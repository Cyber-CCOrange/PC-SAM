## Acknowledgement

This work is built upon the Segment Anything Model (SAM).

- SAM GitHub: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- SAM-B pretrained model used in PC-SAM: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 

# PC-SAM: Patch-Constrained Fine-Grained Interactive Road Segmentation in High-Resolution Remote Sensing Images

This repository provides the implementation of **PC-SAM**, a fine-grained interactive road segmentation model for high-resolution remote sensing images.

PC-SAM supports both **automatic road segmentation** and **interactive segmentation refinement**. By using point prompts, users can correct segmentation errors locally and obtain more accurate road segmentation masks.

---

# SAM/PC-SAM Checkpoint

Before starting, place the pretrained SAM-B checkpoint "sam_vit_b_01ec64.pth" in:

```
./sam-checkpoint/
```

The checkpoints of our PC-SAM can be downloaded from the following link:
- **DeepGlobe**: [PC-SAM_DG.pth](https://drive.google.com/file/d/1NPiLz6NZYU7oCT6_KHJy4pX8C_Bsaxcs/view?usp=sharing)(421.6MB)
- **Massachusetts**: [PC-SAM_MA.pth](https://drive.google.com/file/d/1WB5QABsau9tg6YdRUjbjNDRiSvvI2vQQ/view?usp=sharing)(451.6MB)
- **CHN6-CUG**: [PC-SAM_CH.pth](https://drive.google.com/file/d/12cbocA61KQIzeEcqRJDibwSXrkK1RmUx/view?usp=sharing)(485.1MB)

Place the PC-SAM checkpoints in:

```
./train-checkpoint/
```

---

# Dataset

The dataset should be placed in:

```
./datasets/
```

The dataset directory should be organized as follows:

```
datasets/
└── [dataset_name]/
    ├── input
    ├── gt-mask
    ├── valid-input
    └── valid-mask
```

Where:

- **input**: training images  
- **gt-mask**: ground truth masks for training  
- **valid-input**: validation images  
- **valid-mask**: ground truth masks for validation  

---

# Training

To train **PC-SAM**, run the following command:

```
python PC-SAM.py --config configs/PC-SAM_{Dataset_name}.yml
```

Example: training on the **DeepGlobe dataset**

```
python PC-SAM.py --config configs/PC-SAM_DG.yml
```

---

# Testing

To test the trained model:

```
python PC-SAM.py --config configs/PC-SAM_{Dataset_name}.yml --load_model --valid --batch_size 1
```

Example: testing on the **DeepGlobe dataset**

```
python PC-SAM.py --config configs/PC-SAM_DG.yml --load_model --valid --batch_size 1
```

To save the predictions, add the argument:

```
--save_valid_predictions
```

The predictions will be saved in:

```
./datasets/[dataset_name]/PC-SAM/
```

---

# Visualization

To visualize the segmentation results:

```
python PC-SAM.py --config configs/PC-SAM_{Dataset_name}.yml --load_model --valid --batch_size 1 --valid_show_img_mask
```

Example:

```
python PC-SAM.py --config configs/PC-SAM_DG.yml --load_model --valid --batch_size 1 --valid_show_img_mask
```

---

# Description

PC-SAM is designed for **fine-grained interactive road segmentation** in high-resolution remote sensing images.

The model allows users to refine segmentation results through **interactive point prompts**, enabling local correction of:

- false positives
- missing road segments

This interactive mechanism provides better controllability compared with traditional fully automatic segmentation models.

---

# Citation

If you find this work useful, please cite:

```
@inproceedings{pcsam2026,
title={Patch-Constrained Fine-Grained Interactive Road Segmentation in High-Resolution Remote Sensing Images},
author={XXX},
booktitle={},
year={2026}
}
```

---

# Acknowledgement

This work is built upon the **Segment Anything Model (SAM)** and implemented using **PyTorch**.

---

# License

This project is released for academic research purposes.
