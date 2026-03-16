import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from segment_anything.utils.transforms import ResizeLongestSide
from utils.tensor_process import RandomCropWithLabels, Resize, RandomPatchKeep

class SamDataset(data.Dataset):
    def __init__(self, image_dir, gt_mask_dir, prompt_masks_dir=None):
        self.image_dir = image_dir
        self.gt_mask_dir = gt_mask_dir
        self.prompt_mask_dir = prompt_masks_dir
        self.transform = ResizeLongestSide(1024)

        self.images = [f.replace('_mask.png', '_sat.jpg') for f in os.listdir(gt_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        # 加载和预处理图像
        image = cv2.imread(os.path.join(self.image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.gt_mask_dir, image_name.replace('_sat.jpg', '_mask.png')), 
                         cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        if self.prompt_mask_dir is not None:
            prompt_mask = cv2.imread(os.path.join(self.prompt_mask_dir, image_name.replace('_sat.jpg', '_mask.png')), 
                         cv2.IMREAD_GRAYSCALE)
            prompt_mask = prompt_mask.astype(np.float32) / 255.0
        else:
            prompt_mask = None
        # 图像处理
        original_size = torch.tensor(image.shape[:2])

        # 取出image中非全白部分的mask
        not_white_mask = ~(np.all(image == 255, axis=2))
        not_white_mask = torch.from_numpy(not_white_mask).unsqueeze(0).float()

        input_image = self.transform.apply_image(image)
        # input_image = input_image.astype(np.float32) / 255.0
        # input_image = torch.from_numpy(input_image).permute(2, 0, 1)
        
        # 标准化
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        # input_image = (input_image - mean) / std
        
        # 处理掩码
        mask_torch = torch.from_numpy(mask).float().unsqueeze(0)
        if prompt_mask is not None:
            prompt_mask_torch = torch.from_numpy(prompt_mask).float().unsqueeze(0)
        else:
            prompt_mask_torch = None
        
        return {
            'image': input_image,
            'image_name': image_name,
            'original_size': original_size,
            # 'prompt_mask': prompt_mask_torch,
            'gt_mask': mask_torch,
            'not_white_mask': not_white_mask
        }


class UNetDataset(data.Dataset):
    def __init__(self, image_dir, gt_mask_dir):
        self.image_dir = image_dir
        self.gt_mask_dir = gt_mask_dir
        self.transform = ResizeLongestSide(1024)

        self.images = [f.replace('_mask.png', '_sat.jpg') for f in os.listdir(gt_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        # 加载和预处理图像
        image = cv2.imread(os.path.join(self.image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.gt_mask_dir, image_name.replace('_sat.jpg', '_mask.png')), 
                         cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0

        # 图像处理
        # 取出image中非全白部分的mask
        not_white_mask = ~(np.all(image == 255, axis=2))
        not_white_mask = torch.from_numpy(not_white_mask).unsqueeze(0).float()

        input_image = image
        
        # 处理掩码
        mask_torch = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'image': input_image,
            'image_name': image_name,
            'gt_mask': mask_torch,
            'not_white_mask': not_white_mask
        }



class DataAugment():
    def __init__(self, 
                 scale=[0.5, 2.0], 
                 size=[768, 768], 
                 rotate_degree=[-30, 30], 
                 brightness=0.2, 
                 contrast=0.4, 
                 saturation=0.4, 
                 hue=0.1,
                 device=None):

        self.scale = scale    
        self.size = size
        self.rotate_degree = rotate_degree
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_crop = RandomCropWithLabels((1024, 1024))
        # self.random_crop = RandomPatchKeep((512, 512))
        self.resize = Resize((1536, 1536))
        self.device = device

    def Totensor(self, img):

        transform = transforms.ToTensor()

        return transform(img)

    def Normalize(self, img):
        """
        Use ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
        """
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                         std=[0.229, 0.224, 0.225])
        img = transform(img)
        
        return img

    def Random_Resizecrop(self, img, target):
        """
        Random resize and crop the image and target.

        Args:
            img(torch.Tensor): Input image tensor of shape (B, C, H, W).
            target(torch.Tensor): Input target tensor of shape (B, H, W).
            scale(tuple): Scale range for random resize.
            size(tuple): Random crop size.
        Returns:
            img, target
        """

        r = random.uniform(*self.scale)
        new_h = int(1024 * r)
        new_w = int(2048 * r)
        img = F.resize(img, (new_h, new_w))  
        target = F.resize(target, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)

        pad_h = max(self.size[0] - new_h, 0)
        pad_w = max(self.size[1] - new_w, 0)

        padding = (0, 0, pad_w, pad_h)

        # 对图像填充（图像用 0 填充 → 黑边）
        img = F.pad(img, padding, fill=0)
        target = F.pad(target, padding, fill=255)

        i, j, h, w = transforms.RandomCrop.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        target = F.crop(target, i, j, h, w)

        return img, target

    def Random_HorizontalFlip(self, img, target, nonwhite_mask=None):
        """
        Random horizontal flip the image and target.

        """
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            target = F.hflip(target)
            if nonwhite_mask is not None:
                nonwhite_mask = F.hflip(nonwhite_mask)

        return img, target, nonwhite_mask

    def Random_VerticalFlip(self, img, target, nonwhite_mask=None):
        """
        Random Vertical flip the image and target.

        """
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            target = F.vflip(target)
            if nonwhite_mask is not None:
                nonwhite_mask = F.vflip(nonwhite_mask)

        return img, target, nonwhite_mask   

    def Random_Rotate(self, img, target, nonwhite_mask=None):
        """
        Randomly rotate the image and target by 0, 90, 180, or 270 degrees.
        """
        angles = [0, 90, 180, 270]
        degree = random.choice(angles)
        img = F.rotate(img, degree, interpolation=F.InterpolationMode.BILINEAR, expand=False, fill=0)
        target = F.rotate(target, degree, interpolation=F.InterpolationMode.NEAREST, expand=False, fill=0)
        if nonwhite_mask is not None:
            nonwhite_mask = F.rotate(nonwhite_mask, degree, interpolation=F.InterpolationMode.NEAREST, expand=False, fill=0)
        return img, target, nonwhite_mask
    
    def Random_ColorJitter(self, img):
        """
        Random color jitter the image.

        """
        transform = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        img = transform(img)

        return img
    
    def Random_Erasing_Pair(
        self,
        img,
        mask,
        p=0.5,
        num_patches=(1, 4),
        scale=(0.05, 0.2),
        ratio=(0.3, 3.0),
        erase_value=0.0
    ):
        """
        同步随机擦除（图像与掩码）

        参数:
            img:  (B, C, H, W)
            mask: (B, 1, H, W)
            p: 执行概率
            num_patches: 每张图擦除块数量范围
            scale: 每块擦除面积比例范围
            ratio: 每块宽高比范围
            erase_value: 图像擦除区域填充值
        """
        if torch.rand(1, device=self.device) > p:
            return img, mask

        B, C, H, W = img.shape
        area = H * W
        
        _, _, MH, MW = mask.shape
        mask = F.resize(mask, [H, W], F.InterpolationMode.NEAREST)

        for i in range(B):
            n_patches = torch.randint(num_patches[0], num_patches[1] + 1, (1,), device=self.device).item()
            for _ in range(n_patches):
                target_area = torch.empty(1, device=self.device).uniform_(*scale) * area
                aspect_ratio = torch.empty(1, device=self.device).uniform_(*ratio)

                h = int(torch.sqrt(target_area * aspect_ratio).clamp(1, H - 1))
                w = int(torch.sqrt(target_area / aspect_ratio).clamp(1, W - 1))

                top = int(torch.rand(1, device=self.device) * (H - h))
                left = int(torch.rand(1, device=self.device) * (W - w))

                img[i, :, top:top + h, left:left + w] = erase_value
                mask[i, :, top:top + h, left:left + w] = 0.0

        mask = F.resize(mask, [MH, MW], F.InterpolationMode.NEAREST)

        return img, mask
    
    def train_process(self, img, target, nonwhite_mask=None):
        """
        Process the image(B, H, W, C) and target(B, 1, H, W).

        """
        # img = img[:, :, :, [2, 1, 0]]
        img = img / 255.0
        img = img.permute(0, 3, 1, 2)
        # img = self.Totensor(img)
        img = self.Random_ColorJitter(img)
        img, target, nonwhite_mask = self.Random_HorizontalFlip(img, target, nonwhite_mask)
        img, target, nonwhite_mask = self.Random_VerticalFlip(img, target, nonwhite_mask)
        img, target, nonwhite_mask = self.Random_Rotate(img, target, nonwhite_mask)
        # img, target = self.Random_Erasing_Pair(img, target, p=0.5)
        img = self.Normalize(img)
        
        # img, target, nonwhite_mask = self.random_crop(img, target, nonwhite_mask)
        # img, target = self.resize(img, target)
        
        return img, target, nonwhite_mask
    
    def val_process(self, img, target, is_mass=False):
        """
        Process the image(B, H, W, C) and target(B, 1, H, W).
        
        """
        img = img / 255.0
        img = img.permute(0, 3, 1, 2)
        img = self.Normalize(img)

        if is_mass:
            img, target, _ = self.resize(img, target)

        return img, target
