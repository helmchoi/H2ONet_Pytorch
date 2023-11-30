# [CVPR 2023] H2ONet: Hand-Occlusion-and-Orientation-aware Network for Real-time 3D Hand Mesh Reconstruction

<h4 align = "center">Hao Xu<sup>1,2</sup>, Tianyu Wang<sup>1</sup>, Xiao Tang<sup>1</sup>, Chi-Wing Fu<sup>1,2,3</sup></h4>
<h4 align = "center"> <sup>1</sup>Department of Computer Science and Engineering</center></h4>
<h4 align = "center"> <sup>2</sup>Institute of Medical Intelligence and XR, <sup>3</sup>Shun Hing Institute of Advanced Engineering</center></h4>
<h4 align = "center"> The Chinese University of Hong Kong</center></h4>

This is the official implementation of our CVPR2023 paper [H2ONet](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_H2ONet_Hand-Occlusion-and-Orientation-Aware_Network_for_Real-Time_3D_Hand_Mesh_Reconstruction_CVPR_2023_paper.pdf).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=JN-G8ePC3Mk)].

## Our Poster

![poster](./files/poster.png)

## Todo List

* [X] ~~Single-frame model code~~
* [X] ~~Occlusion label preparation code for the DexYCB dataset~~
* [X] ~~Occlusion label preparation code for the HO3D dataset~~
* [X] ~~Multi-frame model code~~
* [X] ~~Training config details~~
* [X] model checkpoints and evaluation code

## Install

* Environment
  ```
  conda create -n h2onet python=3.8
  conda activate h2onet
  ```
* Requirements
  ```
  pip install -r requirements.txt
  ```
* Download the pre-trained weights of the backbone `densestack.pth` from Google drive or Baidu cloud, which are both provided by [MobRecon](https://github.com/SeanChenxy/HandMesh). After than, put it into the `checkpoints` folder.

## Data Preparation

We evaluate different models on the DexYCB and HO3D datasets. The pre-processed ground truths are from [HandOccNet](https://github.com/namepllet/HandOccNet). Please follow its instruction to prepare the data and ground truths like this,

```
|-- data  
|   |-- HO3D
|   |   |-- train
|   |   |   |-- ABF10
|   |   |   |-- ......
|   |   |-- evaluation
|   |   |-- annotations
|   |   |   |-- HO3D_train_data.json
|   |   |   |-- HO3D_evaluation_data.json
|   |-- DEX_YCB
|   |   |-- 20200709-subject-01
|   |   |-- ......
|   |   |-- annotations
|   |   |   |--DEX_YCB_s0_train_data.json
|   |   |   |--DEX_YCB_s0_test_data.json
```

## Occlusion Label Preparation

Our single-frame model does not need the occlusion prediction. To train the fingle-level occlusion classifier in our multi-frame model, we first prepare the occlusion label from the provided ground truths. Please find more details in `occ_label_preparation`.

## Training

We adopt a two-stage training strategy: (i) hand shape reconstruction at canonical pose; and (ii) hand orientation regression based on the model trained in (i).

For training our single-frame model on the DexYCB dataset,

```
# stage 1
python3 train.py --model_dir=./experiment/single_frame_dexycb/stage_1
# stage 2
python3 train.py --model_dir=./experiment/single_frame_dexycb/stage_2 --resume=./experiment/single_frame_dexycb/stage_1/test_model_best.pth -ow
```

For training our multi-frame model on the DexYCB dataset, we first load the pre-trained single-frame model (stage 1).

```
# stage 1
python3 train.py --model_dir=./experiment/multi_frame_dexycb/stage_1 --resume=./experiment/single_frame_dexycb/stage_1/test_model_best.pth -ow
# stage 2
python3 train.py --model_dir=./experiment/multi_frame_dexycb/stage_2 --resume=./experiment/multi_frame_dexycb/stage_1/test_model_best.pth -ow
```

For training our single-frame and multi-frame models on the HO3D-v2 dataset, we follow the same approach and change the dataset name in the scripts. Note that due to the limited scale of the HO3D-v2 dataset, for training our model in stage 2, we first pre-train it on the DexYCB dataset for a few epochs (<5) to avoid unstable training (e.g., NaN in training).

## Testing

To test our pre-trained model,

```
python test.py --model_dir=./experiment/multi_frame_dexycb/stage_2 --resume=./experiment/multi_frame_dexycb/stage_2/test_model_best.pth
```

We provide the pre-trained multi-frame model on the DexYCB dataset. [[checkpoint](https://drive.google.com/file/d/11VLYLr5bjqCwUgqdihte1hoIcIrkTHt8/view?usp=sharing)]

## Citation

```
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Hao and Wang, Tianyu and Tang, Xiao and Fu, Chi-Wing},
    title     = {H2ONet: Hand-Occlusion-and-Orientation-Aware Network for Real-Time 3D Hand Mesh Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {17048-17058}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [MobRecon](https://github.com/SeanChenxy/HandMesh) (Lightweight pipeline)
* [HandOccNet](https://github.com/namepllet/HandOccNet) (HO3D and DexYCB datasets processing)

We thank the respective authors for open sourcing their methods.
