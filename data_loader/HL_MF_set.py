import logging
import os
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from common.utils.preprocessing import load_img, process_bbox, get_bbox, augmentation_multi_frames
from common.utils.transforms import align_w_scale
from common.utils.mano import MANO

logger = logging.getLogger(__name__)


class HL_MF_set(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split if data_split == "train" else "test"
        self.root_dir = "data/2.Qual"
        self.annot_path = os.path.join(self.root_dir, "annotations")
        self.root_joint_idx = 0
        self.mano = MANO()

        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(os.path.join(self.annot_path, "HL_{}_data.json".format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            img_path = os.path.join(self.root_dir, img["file_name"])
            img_shape = (img["height"], img["width"])

            assert self.data_split != "train"
            
            # hand_type = ann["hand_type"]
            
            bbox = np.array([300.0, 80.0, 660.0, 640.0]).astype(np.float32)
            bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)
            if bbox is None:
                bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

            data = {
                "img_path": img_path,
                "img_shape": img_shape,
                "bbox": bbox,
                "image_id": image_id,
                # "hand_type": hand_type
            }

            datalist.append(data)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data_0 = copy.deepcopy(self.datalist[idx])
        img_path_0, bbox_0 = data_0["img_path"], data_0["bbox"]
        # hand_type = data_0["hand_type"]
        do_flip = False #(hand_type == "left")

        # multi-frame index selection
        img_path_1 = img_path_0[:-5] + "1.png"
        img_path_2 = img_path_0[:-5] + "2.png"

        # img
        img_0 = load_img(img_path_0)
        img_1 = load_img(img_path_1)  # previous short term
        img_2 = load_img(img_path_2)  # previous long term

        img_list = [img_0, img_1, img_2]
        bbox_list = [bbox_0, bbox_0, bbox_0]    # [HL] same
        img_shape_list = [data_0["img_shape"], data_0["img_shape"], data_0["img_shape"]]        # [HL] same
        img_list, img2bb_trans_list, bb2img_trans_list, rot_list, scale = augmentation_multi_frames(img_list=img_list,
                                                                                                    bbox_list=bbox_list,
                                                                                                    data_split=self.data_split,
                                                                                                    input_img_shape=self.cfg.data.input_img_shape,
                                                                                                    scale_factor=0.25,
                                                                                                    rot_factor=30,
                                                                                                    rot_prob=self.cfg.data.rot_prob,
                                                                                                    same_rot=True,
                                                                                                    color_factor=0.2,
                                                                                                    do_flip=do_flip)
        img_list = [self.transform(img.astype(np.float32)) / 255. for img in img_list]
        input = {}
        for img_idx, (img, img2bb_trans, bb2img_trans, rot, img_shape) in enumerate(zip(img_list, img2bb_trans_list, bb2img_trans_list, rot_list, img_shape_list)):
            assert self.data_split != "train"

            input["img_{}".format(img_idx)] = img
            input["img2bb_trans_{}".format(img_idx)] = img2bb_trans
            input["bb2img_trans_{}".format(img_idx)] = bb2img_trans
        
        # For saving debugging images
        input["img_path"] = img_path_0
        input["bbox"] = bbox_0

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        print("[Error] Not implemented for HL dataset")
        return None

    def print_eval_result(self, test_epoch):
        print("[Error] Not implemented for HL dataset")
