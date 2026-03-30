# PC-SAM: Patch-Constrained Fine-Grained Interactive Road Segmentation in High-Resolution Remote Sensing Images

This repository provides the implementation of **PC-SAM**, a fine-grained interactive road segmentation model for high-resolution remote sensing images.

PC-SAM supports both **automatic road segmentation** and **interactive segmentation refinement**. By using point prompts, users can correct segmentation errors locally and obtain more accurate road segmentation masks.

---

# SAM Checkpoint

Put pretrained SAM-B model "sam_vit_b_01ec64.pth" in:

```
./sam-checkpoint/
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
