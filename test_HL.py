import argparse
import os
import torch
import json
from tqdm import tqdm
from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config
from common.utils.preprocessing import generate_patch_image
import numpy as np
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--resume", default="", type=str, help="Path of model weights")
parser.add_argument("--out_dir", default="", type=str, help="Output directory")


def test(model, mng: Manager, out_dir):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()

    if not os.path.exists(out_dir + "/result"):
        os.makedirs(out_dir + "/result")
    if not os.path.exists(out_dir + "/vis"):
        os.makedirs(out_dir + "/vis")

    result_stack = []
    pred_joints3d_w_gr = []
    pred_joints2d_w_gr = []

    with torch.no_grad():
        # Compute metrics over the dataset <-- Only Test set!
        split = "test"
        assert split in mng.dataloader
        
        # Use tqdm for progress bar
        t = tqdm(total=len(mng.dataloader[split]))
        fps = 0.0
        # cur_sample_idx = 0
        for batch_idx, batch_input in enumerate(mng.dataloader[split]):
            # Move data to GPU if available
            batch_input = tool.tensor_gpu(batch_input)

            # Compute model output
            t0 = time.time()
            batch_output = model(batch_input)
            t1 = time.time()
            # print("==> inference time: ", t1-t0, ", fps: ", 1/(t1-t0))
            fps += 1.0 / (t1 - t0)

            # Get real batch size
            if "img" in batch_input:
                batch_size = batch_input["img"].size()[0]
            elif "img_0" in batch_input:
                batch_size = batch_input["img_0"].size()[0]
            else:
                batch_size = mng.cfg.test.batch_size

            batch_output = tool.tensor_gpu(batch_output, check_on=False)
            
            result_stack += [{k: v[bid].tolist() for k, v in batch_output.items()} for bid in range(batch_size)]
            
            batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]

            for dict_idx, dict_ in enumerate(batch_output):
                # 3D_w_gr
                pred_joints3d_now = dict_['pred_joints3d_w_gr'].tolist()
                pred_joints3d = []
                for joints in pred_joints3d_now:
                    pred_joints3d += joints
                pred_joints3d_w_gr += [pred_joints3d]
                # 2D
                pred_joints2d_now = dict_['pred_joints_img'].tolist()
                pred_joints2d = []
                for joints in pred_joints2d_now:
                    joints_ = [joints[0] * mng.cfg.data.input_img_shape[1], joints[1] * mng.cfg.data.input_img_shape[0]]
                    if "img" in batch_input:
                        bb2img_trans = batch_input["bb2img_trans"].cpu().numpy()[0]
                    elif "img_0" in batch_input:
                        bb2img_trans = batch_input["bb2img_trans_0"].cpu().numpy()[0]
                    joints__ = np.dot(bb2img_trans, np.array([joints_[0], joints_[1], 1.0]))
                    pred_joints2d += [joints__[0], joints__[1]]
                pred_joints2d_w_gr += [pred_joints2d]                

            # Save 2D joint on images for vis.
            for imgidx, imgpath in enumerate(batch_input["img_path"]):
                img_ = cv2.imread(imgpath)
                bbox_ = batch_input["bbox"][imgidx]
                img_patch, _, _ = generate_patch_image(img_, bbox_, 1.0, 0.0, False, mng.dataset[split].cfg.data.input_img_shape)
                for jnt in batch_output[imgidx]["pred_joints_img"]:
                    img_patch = cv2.circle(img_patch, (int(jnt[0] * np.shape(img_patch)[0]), int(jnt[1] * np.shape(img_patch)[1])),
                                           3, (0,0,255), 3)
                cv2.imwrite(out_dir + "/vis/" + imgpath[29:], img_patch)

            # # evaluate
            # metric = mng.dataset[split].evaluate(batch_output, cur_sample_idx)
            # cur_sample_idx += len(batch_output)
            # if "DEX" in mng.cfg.data.name:
            #     mng.update_metric_status(metric, split, batch_size)

            # Tqdm settings
            t.set_description(desc="")
            t.update()

        t.close()
        fps /= float(len(mng.dataloader[split]))
        print("** Mean fps: ", fps)
    
    with open(out_dir + "/result/output.json", "w") as f:
        json.dump(result_stack, f)
    with open(out_dir + "/result/pred_joints3d.json", "w") as f:
        json.dump(pred_joints3d_w_gr, f)
    with open(out_dir + "/result/pred_joints2d_img.json", "w") as f:
        json.dump(pred_joints2d_w_gr, f)


def main(cfg, out_dir):
    # Set rank and is_master flag
    cfg.base.only_weights = False
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.base.model_dir, "test.log"))
    # Print GPU ids
    gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    logger.info("Using GPU ids: [{}]".format(gpu_ids))
    # Fetch dataloader
    cfg.data.eval_type = ["test"]
    dl, ds = fetch_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg)
    # Initialize manager
    mng = Manager(model=model, optimizer=None, scheduler=None, cfg=cfg, dataloader=dl, dataset=ds, logger=logger)
    # Test the model
    mng.logger.info("Starting test.")
    # Load weights from restore_file if specified
    if mng.cfg.base.resume is not None:
        mng.load_ckpt()
    test(model, mng, out_dir)


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg
    # Update args into cfg.base
    cfg.base.update(vars(args))
    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    main(cfg, args.out_dir)
