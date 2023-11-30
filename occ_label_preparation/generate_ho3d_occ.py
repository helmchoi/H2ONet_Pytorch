import os
import numpy as np
import cv2
import torch
import copy
from pycocotools.coco import COCO
import json
from common.utils.preprocessing import load_img, process_bbox, get_bbox
from common.utils.vis import vis_mesh, render_mesh_seg
from common.utils.mano import MANO
from common.utils.transforms import cam2pixel
from occ_label_preparation.seg import *

mano = MANO()

def load_ho3d_data(data_split):
    root_dir = "data/HO3D_v2"
    annot_path = os.path.join(root_dir, "annotations")
    data_split_for_load = "train" if data_split == "train" or data_split == "val" else "evaluation"
    db = COCO(os.path.join(annot_path, "HO3D_{}_data.json".format(data_split_for_load)))

    datalist = []
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann["image_id"]
        img = db.loadImgs(image_id)[0]
        img_path = os.path.join(root_dir, data_split_for_load, img["file_name"])
        img_shape = (img["height"], img["width"])
        if data_split == "train" or data_split == "val":
            joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)  # meter
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
            joints_coord_img = cam2pixel(joints_coord_cam, cam_param["focal"], cam_param["princpt"])
            bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
            bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)
            if bbox is None:
                continue
            bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
            bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
            bbox_size_x = bbox[2] / img["width"]
            bbox_size_y = bbox[3] / img["height"]
            bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size_x, bbox_size_y])

            mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
            mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

            data = {
                "img_path": img_path,
                "img_shape": img_shape,
                "joints_coord_cam": joints_coord_cam,
                "joints_coord_img": joints_coord_img,
                "bbox": bbox,
                "bbox_pos": bbox_pos,
                "cam_param": cam_param,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape
            }
        else:
            root_joint_cam = np.array(ann["root_joint_cam"], dtype=np.float32)
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
            bbox = np.array(ann["bbox"], dtype=np.float32)
            bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.5)
            bbox_center_x = (bbox[0] + bbox[2]) / 2.0 - (img["width"] / 2.0)
            bbox_center_y = (bbox[1] + bbox[3]) / 2.0 - (img["height"] / 2.0)
            bbox_size_x = bbox[2] / img["width"]
            bbox_size_y = bbox[3] / img["height"]
            bbox_pos = np.array([bbox_center_x, bbox_center_y, bbox_size_x, bbox_size_y])

            data = {
                "img_path": img_path,
                "img_shape": img_shape,
                "root_joint_cam": root_joint_cam,
                "bbox": bbox,
                "bbox_pos": bbox_pos,
                "cam_param": cam_param,
            }

        datalist.append(data)
    return datalist


def get_fingers_occ_label(gt_verts_pixel, render_seg, img_shape):
    # occ
    w_out_of_range_mask = gt_verts_pixel[:, 0] >= img_shape[1]
    h_out_of_range_mask = gt_verts_pixel[:, 1] >= img_shape[0]
    out_of_range_mask = np.logical_or(w_out_of_range_mask, h_out_of_range_mask)
    gt_verts_pixel[out_of_range_mask] = 0
    gt_verts_pixel = gt_verts_pixel.astype(np.int32)

    dmz_vertex_pixel = gt_verts_pixel[thumb_verts_idx]
    dmz_occ_mask = np.all(render_seg[dmz_vertex_pixel[:, 1], dmz_vertex_pixel[:, 0]] == dmz_color, axis=1)

    sz_vertex_pixel = gt_verts_pixel[index_verts_idx]
    sz_occ_mask = np.all(render_seg[sz_vertex_pixel[:, 1], sz_vertex_pixel[:, 0]] == sz_color, axis=1)

    zz_vertex_pixel = gt_verts_pixel[middle_verts_idx]
    zz_occ_mask = np.all(render_seg[zz_vertex_pixel[:, 1], zz_vertex_pixel[:, 0]] == zz_color, axis=1)

    wmz_vertex_pixel = gt_verts_pixel[ring_verts_idx]
    wmz_occ_mask = np.all(render_seg[wmz_vertex_pixel[:, 1], wmz_vertex_pixel[:, 0]] == wmz_color, axis=1)

    xmz_vertex_pixel = gt_verts_pixel[little_verts_idx]
    xmz_occ_mask = np.all(render_seg[xmz_vertex_pixel[:, 1], xmz_vertex_pixel[:, 0]] == xmz_color, axis=1)

    palm_vertex_pixel = gt_verts_pixel[palm_verts_idx]
    palm_occ_mask = np.all(render_seg[palm_vertex_pixel[:, 1], palm_vertex_pixel[:, 0]] == palm_color, axis=1)

    finger_names = ["thumb", "index", "middle", "ring", "little", "palm"]
    occ_mask_list = [thumb_occ_mask, index_occ_mask, middle_occ_mask, ring_occ_mask, little_occ_mask, palm_occ_mask]
    occ_label_list = []
    occ_ratio_list = []
    occ_count_list = []
    for finger_name, occ_mask in zip(finger_names, occ_mask_list):
        occ_ratio_list.append(round(occ_mask.sum() / occ_mask.shape[0], 2))
        occ_label_list.append(int(occ_mask.sum() < 30))
        occ_count_list.append(occ_mask.sum())
    return occ_label_list, occ_ratio_list, occ_count_list


def generate_ho3d_occ_gt():
    data_split = "train"
    datalist = load_ho3d_data(data_split)

    for idx, data in enumerate(datalist):
        img_path, img_shape, bbox = data["img_path"], data["img_shape"], data["bbox"]
        scene_name = img_path.split("/")[-3]
        file_name = img_path.split("/")[-1]
        frame_idx = int(file_name.split(".")[0])
        
        # occ label output filepath
        occ_info_ouput_path = img_path.replace("rgb", "occ").replace("png", "json")
        if not os.path.exists(os.path.dirname(occ_info_ouput_path)):
            os.makedirs(os.path.dirname(occ_info_ouput_path))

        # seg
        seg_path = img_path.replace("rgb", "seg").replace("png", "jpg")
        seg = load_img(seg_path)[:, :, 0] > 200
        seg = cv2.resize(seg.astype(np.float32), (640, 480), cv2.INTER_NEAREST)
        seg = np.tile(seg[:, :, None], (1, 1, 3))

        # gt joints and verts
        gt_mano_pose = torch.from_numpy(data["mano_pose"][None, ...])
        gt_mano_shape = torch.from_numpy(data["mano_shape"][None, ...])
        gt_verts, gt_joints = mano.layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
        gt_verts = gt_verts.squeeze().cpu().numpy()
        gt_joints = gt_joints.squeeze().cpu().numpy()
        gt_verts = gt_verts / 1000.
        gt_joints = gt_joints / 1000.

        # occlusion mask
        root_joint_idx = 0
        gt_verts = gt_verts - gt_joints[0] + data["joints_coord_cam"][root_joint_idx]
        gt_verts_pixel = cam2pixel(gt_verts, data["cam_param"]["focal"], data["cam_param"]["princpt"])
        
        # img
        img = load_img(img_path)  # current
        ori_img = copy.deepcopy(img)
        
        # render predicted mesh for input hand image
        render_seg_ori, render_mask_ori = render_mesh_seg(ori_img, gt_verts, mano.face, data["cam_param"])
        render_seg = seg * render_seg_ori + (1 - seg) * ori_img
        occ_label, occ_ratio, occ_count_list = get_fingers_occ_label(gt_verts_pixel, render_seg, img.shape)
        
        # global occ
        global_occ_ratio = np.sum(seg[:, :, 0]) / np.sum(render_mask_ori[:, :, 0])
        occ_label.append(int(global_occ_ratio < 0.40))
        occ_ratio.append(round(global_occ_ratio, 2))
        occ_count_list.append(0)
        
        # print info
        occ_label_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_label)
        occ_ratio_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_ratio)
        occ_count_str = "Th: {}, Ind: {}, Mid: {}, Ring: {}, Lit: {}, Palm: {}, Glob: {}".format(*occ_count_list)
        
        # dump json
        occ_info_dict = {
            "thumb": occ_label[0],
            "index": occ_label[1],
            "middle": occ_label[2],
            "ring": occ_label[3],
            "little": occ_label[4],
            "palm": occ_label[5],
            "global": occ_label[6],
        }
        with open(occ_info_ouput_path, "w") as f:
            json.dump(occ_info_dict, f)

        # save intermediate results
        if idx % 50 == 0:
            mesh_ori = vis_mesh(ori_img, gt_verts_pixel)
            render_seg = cv2.putText(render_seg, "{}".format(frame_idx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_label_str), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_count_str), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            render_seg = cv2.putText(render_seg, "{}".format(occ_ratio_str), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            img = cv2.putText(img, "{}".format(frame_idx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cat_img = np.concatenate([img, render_seg_ori, seg * 255, render_seg, mesh_ori], axis=1)
            print("writing {}/{}".format(scene_name, frame_idx))
            os.makedirs("demo/ho3d.occ_gt_images/{}".format(scene_name), exist_ok=True)
            cv2.imwrite("demo/ho3d.occ_gt_images/{}/{}".format(scene_name, file_name), cat_img[:, :, ::-1])


if __name__ == "__main__":
    generate_ho3d_occ_gt()
