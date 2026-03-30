import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F

class RandomCropWithLabels:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, nonwhite_mask=None):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        if nonwhite_mask is not None:
            nonwhite_mask = F.crop(nonwhite_mask, i, j, h, w)
            return image, label, nonwhite_mask
        else:
            return image, label

class Resize:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, image, label, nonwhite_mask=None):
        image = F.resize(image, self.target_size, interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, self.target_size, interpolation=F.InterpolationMode.NEAREST)
        if nonwhite_mask is not None:
            nonwhite_mask = F.resize(nonwhite_mask, self.target_size, interpolation=F.InterpolationMode.NEAREST)

        return image, label, nonwhite_mask

def DeNormalize(image, mean, std):
    """
    Denormalize an image tensor.
    
    Args:
        image (torch.Tensor): Image tensor to denormalize.
        mean (tuple): Mean values for each channel.
        std (tuple): Standard deviation values for each channel.
    
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
    return image * std + mean

class RandomPatchKeep:
    def __init__(self, patch_size):
        self.patch_size = patch_size  # (h, w)

    def __call__(self, image, label):
        # 原图尺寸
        assert torch.is_tensor(image)
        H, W = image.size()[-2:]
        ph, pw = self.patch_size

        # 随机选择patch左上角坐标
        top = torch.randint(0, H - ph + 1, (1,)).item()
        left = torch.randint(0, W - pw + 1, (1,)).item()

        # 裁出patch
        image_patch = F.crop(image, top, left, ph, pw)
        label_patch = F.crop(label, top, left, ph, pw)

        # 构造全 0 图（同原图大小）
        image_new = torch.zeros_like(image)
        label_new = torch.zeros_like(label)

        # 把patch放回原位置
        image_new[..., top:top+ph, left:left+pw] = image_patch
        label_new[..., top:top+ph, left:left+pw] = label_patch

        return image_new, label_new