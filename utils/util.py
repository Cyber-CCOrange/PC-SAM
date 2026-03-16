from ast import Tuple
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything.utils.transforms import ResizeLongestSide
from typing import Tuple

def create_point_labels(mask: torch.Tensor, num_points: int, fore_rate: float):
    """
    Input: 
        mask: shape:(B, H, W) or (B, 1, H, W)
    
    Return:
        fore: (fore points: (B, num fore, 2), fore label: (B, num_fore))
        back: (back points: (B, num back, 2), back label: (B, num_back))
    """

    if len(mask.shape) != 3:
        mask= mask.squeeze(1)

    device = mask.device
    B = mask.shape[0]
    num_fore = int(num_points * fore_rate)
    num_back = num_points - num_fore
    fps = []
    bps = []
    fl = []
    bl = []
    for i in range(B):
        # sample foreground/background points (may be fewer than requested)
        fore_points = random_fore_points(mask[i], num_fore)  # [<=num_fore, 2]
        back_points = random_fore_points(mask[i], num_back)  # [<=num_back, 2]
        # back_points = random_fore_points((1 - mask[i]), num_back)  # [<=num_back, 2]

        # compute missing counts
        fore_missing = max(0, num_fore - fore_points.shape[0])
        back_missing = max(0, num_back - back_points.shape[0])

        # pad points to fixed lengths with sentinel (-1,-1)
        if fore_missing > 0:
            pad_fp = torch.full((fore_missing, 2), -1, device=device, dtype=fore_points.dtype)
            fore_points = torch.cat([fore_points, pad_fp], dim=0)
        if back_missing > 0:
            pad_bp = torch.full((back_missing, 2), -1, device=device, dtype=back_points.dtype)
            back_points = torch.cat([back_points, pad_bp], dim=0)

        # build labels aligned with points: 1 for fg, 0 for bg, -1 for padded
        if fore_missing > 0:
            fore_label = torch.cat([
                torch.ones((num_fore - fore_missing), device=device),
                -1 * torch.ones((fore_missing), device=device)
            ], dim=0)
        else:
            fore_label = torch.ones((num_fore), device=device)

        if back_missing > 0:
            back_label = torch.cat([
                torch.zeros((num_back - back_missing), device=device),
                -1 * torch.ones((back_missing), device=device)
            ], dim=0)
        else:
            back_label = torch.zeros((num_back), device=device)

        # collect
        fl.append(fore_label)
        bl.append(back_label)
        fps.append(fore_points)
        bps.append(back_points)

    # stack to fixed shapes
    fore_points = torch.stack(fps)              # [B, num_fore, 2]
    back_points = torch.stack(bps)              # [B, num_back, 2]
    fore_labels = torch.stack(fl)               # [B, num_fore]
    back_labels = torch.stack(bl)               # [B, num_back]

    fore = (fore_points, fore_labels)
    back = (back_points, back_labels)

    return fore, back

def random_fore_points(mask: torch.Tensor, num_points: int):
    """
    Input: 
        mask: shape: (H, W)

    Return:
        points: shape: (N, 2)
    """
    coords = torch.nonzero(mask, as_tuple=False)  # [num_nonzero, 2]

    num_fore_points = coords.shape[0]
    perm = torch.randperm(num_fore_points)

    fore_points = torch.flip(coords[perm[:num_points]], dims=[1])

    return fore_points

def points_transform(points: Tuple[torch.Tensor, torch.Tensor], original_size: Tuple[int, int], target_size: int):
    """
    Input:
        points: (points: (B, N, 2), labels: (B, N))
        original_size: Tuple[int, int]
        target_size: int
    Return:
        points: (points: (B, N, 2), labels: (B, N))
    """

    coords, labels = points
    B = coords.shape[0]
    device = coords.device
    transform = ResizeLongestSide(target_size)

    if type(coords) is not np.ndarray:
        coords = coords.cpu().numpy()

    new_coords = []
    for b in range(B):
        new_coords_per_batch = transform.apply_coords(coords[b], original_size)
        new_coords_torch = torch.as_tensor(new_coords_per_batch, dtype=torch.float, device=device)
        new_coords.append(new_coords_torch)
    new_coords = torch.stack(new_coords, dim=0)
    new_points = (new_coords, labels)

    return new_points

def show_img_mask(
    img: torch.Tensor = None, 
    gt_mask: torch.Tensor = None, 
    logits: torch.Tensor = None,
    repaired_part_logits: torch.Tensor = None, 
    prompt_logits: torch.Tensor = None,
    fore_points: torch.Tensor = None,
    back_points: torch.Tensor = None,
    threshold: float = None):
    """
    Input:
        img: shape: (3, H, W)
        gt_mask: shape: (1, H, W)
        logits: shape: (1, H, W)
        repaired_part_logits: shape: (1, H, W)
        prompt_logits: shape: (1, H, W)
        fore_points: shape: (N, 2)
        back_points: shape: (N, 2)
    
    Return:
        image of mask
    """
    
    logits = logits.squeeze(0)    # H, W
    mask_bin = (logits > threshold).float() * 255  # H, W
    mask_bin = draw_points(mask_bin, back_points, color=(255, 0, 0), point_size=13) / 255

    repaired_part_logits = repaired_part_logits.squeeze(0)    # H, W
    repaired_part_mask_bin = (repaired_part_logits > threshold).float() * 255  # H, W
    repaired_part_mask_bin = draw_points(repaired_part_mask_bin, fore_points, color=(0, 255, 0), point_size=13) / 255

    prompt_logits = prompt_logits.squeeze(0)
    # model_mask = (model_logits > threshold).float()
    model_mask = (((repaired_part_logits > threshold).float() * 255 + (logits > threshold).float() * 255) > threshold).float()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean

    img = img.permute(1, 2, 0).numpy()  # H, W, 3
    gt_mask = gt_mask.squeeze(0).numpy()    # H, W

    fig = plt.figure('img map', figsize=(5, 5), dpi=500)
    img_fig = fig.add_subplot(2,4,1)
    img_fig.imshow(img)
    img_fig.set_title('RGB image', fontsize=3, pad=1)
    plt.axis('off')
    logits_fig = fig.add_subplot(2,4,2)
    logits_fig.imshow(logits, cmap="gray")
    logits_fig.set_title('AMD logits', fontsize=3, pad=1)
    plt.axis('off')
    repaired_part_logits_fig = fig.add_subplot(2,4,3)
    repaired_part_logits_fig.imshow(repaired_part_logits, cmap="gray")
    repaired_part_logits_fig.set_title('PMD logits', fontsize=3, pad=1)
    plt.axis('off')
    model_logits_fig = fig.add_subplot(2,4,4)
    model_logits_fig.imshow(prompt_logits, cmap="gray")
    model_logits_fig.set_title('HRMD logits', fontsize=3, pad=1)
    plt.axis('off')
    gt_mask_fig = fig.add_subplot(2,4,5)
    gt_mask_fig.imshow(gt_mask, cmap="gray")
    gt_mask_fig.set_title('GT mask', fontsize=3, pad=1)
    plt.axis('off')
    mask_fig = fig.add_subplot(2,4,6)
    mask_fig.imshow(mask_bin, cmap="gray")
    mask_fig.set_title('AMD mask', fontsize=3, pad=1)
    plt.axis('off')
    repaired_part_mask_fig = fig.add_subplot(2,4,7)
    repaired_part_mask_fig.imshow(repaired_part_mask_bin, cmap="gray")
    repaired_part_mask_fig.set_title('PMD mask', fontsize=3, pad=1)
    plt.axis('off')
    model_mask_fig = fig.add_subplot(2,4,8)
    model_mask_fig.imshow(model_mask, cmap="gray")
    model_mask_fig.set_title('Two mask', fontsize=3, pad=1)
    plt.axis('off')
    fig.show()
    
# 对 (B, C, H, W) tensor 进行可用于梯度回传的开运算（先腐蚀后膨胀），直接使用 pytorch 实现
def differentiable_opening(input_tensor, erosion_kernel_size=3, dilation_kernel_size=3):
    """
    对输入的 (B, C, H, W) tensor 做可微分的形态学开运算（腐蚀-膨胀）
    Args:
        input_tensor: (B, C, H, W), tensor, 一般在 [0,1] 域
        kernel_size: 卷积核大小，默认3
        padding_mode: 边界填充方式
    Returns:
        opened tensor, (B, C, H, W)
    """
    ero_pad = erosion_kernel_size // 2
    dil_pad = dilation_kernel_size // 2
    # first erosion (min pooling)
    neg_x = -input_tensor
    eroded = -F.max_pool2d(neg_x, erosion_kernel_size, stride=1, padding=ero_pad)
    # then dilation (max pooling)
    opened = F.max_pool2d(eroded, dilation_kernel_size, stride=1, padding=dil_pad)
    return opened

    # Example usage:
    # output = differentiable_opening(input_tensor, kernel_size=3)



def get_randomized_prompt_parameters(base_num_prompt_points, base_fore_rate, prompt_points_delta, fore_rate_delta):
    # num_prompt_points: 基础值±num_prompt_points_delta
    num_prompt_points_delta = int(base_num_prompt_points * prompt_points_delta)
    npp = base_num_prompt_points + np.random.randint(-num_prompt_points_delta, num_prompt_points_delta+1)
    npp = max(0, npp)
    # fore_rate: 基础值±fore_rate_delta，限制在[0.00, 1.0]
    fr = base_fore_rate + np.random.uniform(-fore_rate_delta, fore_rate_delta)
    fr = float(np.clip(fr, 0.00, 1.0))
    return npp, fr

def show_tensor_image(tensor, title="", is_mask=False):
    # tensor shape: (B, C, H, W) or (B, 1, H, W)
    np_img = tensor.detach().cpu().numpy()
    for i in range(min(1, np_img.shape[0])):
        img = np_img[i]
        if img.shape[0] == 1 or is_mask:
            img = img.squeeze(0)
            plt.imshow(img, cmap='gray')
        else:
            # Assume image has 3 channels
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean  # 反归一化
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

def generate_patch_labels_from_points(gt_mask: torch.Tensor, fore: tuple, back: tuple, patch_size: int = 32):
    """
    根据create_point_labels生成的正负点，生成两张标签图
    
    Input:
        gt_mask: shape (B, H, W) 或 (B, 1, H, W)
        fore: (fore_points: (B, num_fore, 2), fore_labels: (B, num_fore))
        back: (back_points: (B, num_back, 2), back_labels: (B, num_back))
        patch_size: patch大小，默认32
    
    Return:
        pos_label_map: (B, H, W) - 正标签图，只保留有正点的patch区域
        neg_label_map: (B, H, W) - 负标签图，去掉gt_mask上有负点的patch区域
    """
    # 处理mask维度
    if len(gt_mask.shape) == 4:
        gt_mask = gt_mask.squeeze(1)  # (B, H, W)
    
    device = gt_mask.device
    B, H, W = gt_mask.shape
    
    fore_points, fore_labels = fore  # fore_points: (B, num_fore, 2), fore_labels: (B, num_fore)
    back_points, back_labels = back  # back_points: (B, num_back, 2), back_labels: (B, num_back)
    
    # 初始化标签图
    pos_label_map = torch.zeros_like(gt_mask)  # 正标签图，初始全0
    neg_label_map = gt_mask.clone()  # 负标签图，初始为gt_mask的副本
    
    for b in range(B):
        # 获取当前batch的有效正点（label != -1）
        valid_fore_mask = fore_labels[b] != -1
        valid_fore_points = fore_points[b][valid_fore_mask]  # [num_valid_fore, 2]
        
        # 获取当前batch的有效负点（label != -1）
        valid_back_mask = back_labels[b] != -1
        valid_back_points = back_points[b][valid_back_mask]  # [num_valid_back, 2]
        
        # 计算patch网格大小
        num_patches_h = (H + patch_size - 1) // patch_size
        num_patches_w = (W + patch_size - 1) // patch_size
        
        # 标记包含正点的patch
        pos_patch_mask = torch.zeros((num_patches_h, num_patches_w), device=device, dtype=torch.bool)
        if valid_fore_points.shape[0] > 0:
            # 点的格式是 (x, y)，即 (列, 行)
            # 计算每个正点所在的patch索引
            patch_rows = (valid_fore_points[:, 1] // patch_size).long()  # y坐标对应行
            patch_cols = (valid_fore_points[:, 0] // patch_size).long()  # x坐标对应列
            
            # 确保索引在有效范围内
            patch_rows = torch.clamp(patch_rows, 0, num_patches_h - 1)
            patch_cols = torch.clamp(patch_cols, 0, num_patches_w - 1)
            
            # 标记包含正点的patch
            pos_patch_mask[patch_rows, patch_cols] = True
        
        # 标记包含负点的patch
        neg_patch_mask = torch.zeros((num_patches_h, num_patches_w), device=device, dtype=torch.bool)
        if valid_back_points.shape[0] > 0:
            # 计算每个负点所在的patch索引
            patch_rows = (valid_back_points[:, 1] // patch_size).long()  # y坐标对应行
            patch_cols = (valid_back_points[:, 0] // patch_size).long()  # x坐标对应列
            
            # 确保索引在有效范围内
            patch_rows = torch.clamp(patch_rows, 0, num_patches_h - 1)
            patch_cols = torch.clamp(patch_cols, 0, num_patches_w - 1)
            
            # 标记包含负点的patch
            neg_patch_mask[patch_rows, patch_cols] = True
        
        # 生成正标签图：只保留有正点的patch区域
        for ph in range(num_patches_h):
            for pw in range(num_patches_w):
                if pos_patch_mask[ph, pw]:
                    h_start = ph * patch_size
                    h_end = min(h_start + patch_size, H)
                    w_start = pw * patch_size
                    w_end = min(w_start + patch_size, W)
                    # 保留该patch区域的gt_mask值
                    pos_label_map[b, h_start:h_end, w_start:w_end] = gt_mask[b, h_start:h_end, w_start:w_end]
        
        # 生成负标签图：去掉有负点的patch区域
        for ph in range(num_patches_h):
            for pw in range(num_patches_w):
                if neg_patch_mask[ph, pw]:
                    h_start = ph * patch_size
                    h_end = min(h_start + patch_size, H)
                    w_start = pw * patch_size
                    w_end = min(w_start + patch_size, W)
                    # 将该patch区域置0
                    neg_label_map[b, h_start:h_end, w_start:w_end] = 0
    
    return pos_label_map, neg_label_map

def draw_points(
    mask: np.ndarray,
    points: np.ndarray,
    color: tuple = (0, 255, 0),
    point_size: int = 7,
    neg_points: np.ndarray = None,
    neg_color: tuple = (255, 0, 0),
):
    """
    在mask上标注点
    
    Input:
        mask: shape (H, W) - 单通道mask，值域[0, 255]
        points: shape (N, 2) - 点坐标，格式为(x, y)，即(列, 行)
        color: tuple (R, G, B) - 点的颜色，默认红色(255, 0, 0)
        point_size: int - 点的大小（半径），默认3，即3x3的方块
    
    Return:
        mask_rgb: shape (H, W, 3) - RGB格式的mask，标注了点
    """
    # 转换为RGB
    if len(mask.shape) == 2:
        mask_rgb = np.stack([mask, mask, mask], axis=-1)  # H, W, 3
    else:
        mask_rgb = mask.copy()
    
    # 标注正点
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < mask_rgb.shape[1] and 0 <= y < mask_rgb.shape[0]:
            radius = point_size // 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask_rgb.shape[0] and 0 <= nx < mask_rgb.shape[1]:
                        mask_rgb[ny, nx] = color

    # 额外标注负点（如果提供）
    if neg_points is not None:
        for point in neg_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < mask_rgb.shape[1] and 0 <= y < mask_rgb.shape[0]:
                radius = point_size // 2
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < mask_rgb.shape[0] and 0 <= nx < mask_rgb.shape[1]:
                            mask_rgb[ny, nx] = neg_color
    
    return mask_rgb

def save_masks_with_points(input_image: torch.Tensor,
                           gt_mask: torch.Tensor,
                           mask_bin: torch.Tensor, 
                           repaired_part_mask_bin: torch.Tensor,
                           fore_points: tuple,
                           back_points: tuple,
                           output_dir: Path,
                           input_image_name: list,
                           draw_points_on_mask: bool = True):
    """
    保存mask并可选地标注正负点
    
    Input:
        mask_bin: shape (B, H, W) - 预测的mask，值域[0, 1]
        repaired_part_mask_bin: shape (B, H, W) - 修复部分的mask，值域[0, 1]
        fore_points: tuple (fore_points_tensor: (B, num_fore, 2), fore_labels: (B, num_fore)) - 前景点
        back_points: tuple (back_points_tensor: (B, num_back, 2), back_labels: (B, num_back)) - 背景点
        output_dir: Path - 输出目录
        input_image_name: list - 图像名称列表，长度为B
        draw_points_on_mask: bool - 是否在mask上标注点，默认True
    
    Return:
        None
    """
    fore_points_tensor, fore_labels_tensor = fore_points
    back_points_tensor, back_labels_tensor = back_points
    
    B = mask_bin.shape[0]
    
    for i in range(B):
        # 过滤有效点（label != -1）
        valid_fore_mask = fore_labels_tensor[i] != -1
        valid_back_mask = back_labels_tensor[i] != -1
        
        valid_fore_points = fore_points_tensor[i][valid_fore_mask].cpu().numpy()  # [num_valid_fore, 2]
        valid_back_points = back_points_tensor[i][valid_back_mask].cpu().numpy()  # [num_valid_back, 2]
        
        image = (input_image[i] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        im = Image.fromarray(image)
        im.save(output_dir / f"{input_image_name[i].replace('.jpg', '_mask_0.png')}")

        if len(gt_mask.shape) == 4:
            gt_mask = gt_mask.squeeze(1)
        gt = gt_mask[i].cpu().numpy().astype(np.uint8) * 255
        im = Image.fromarray(gt)
        im.save(output_dir / f"{input_image_name[i].replace('.jpg', '_mask_1.png')}")

        # 保存mask_output，标注负点（红色）
        mask_output = mask_bin[i].cpu().numpy().astype(np.uint8) * 255
        if draw_points_on_mask:
            mask_output_rgb = draw_points(mask_output, valid_back_points, color=(255, 0, 0), point_size=13)
        else:
            mask_output_rgb = np.stack([mask_output, mask_output, mask_output], axis=-1)
        
        im = Image.fromarray(mask_output_rgb)
        im.save(output_dir / f"{input_image_name[i].replace('.jpg', '_mask_2.png')}")
        
        # 保存repaired_output，标注正点（绿色）
        repaired_output = repaired_part_mask_bin[i].cpu().numpy().astype(np.uint8) * 255
        if draw_points_on_mask:
            repaired_output_rgb = draw_points(repaired_output, valid_fore_points, color=(0, 255, 0), point_size=13)
        else:
            repaired_output_rgb = np.stack([repaired_output, repaired_output, repaired_output], axis=-1)
        
        im = Image.fromarray(repaired_output_rgb)
        im.save(output_dir / f"{input_image_name[i].replace('.jpg', '_mask_3.png')}")

def sample_points_from_patches(mask: torch.Tensor, patch_size: int = 32, sample_per_patch: int = 1, point_label: int = 1):
    """
    将mask均分成patch，在每个patch上从mask的前景区域随机采样点，构成SAM的point prompt输入
    
    Input:
        mask: shape (B, H, W) 或 (B, 1, H, W) - 输入mask，值域[0, 1]
        patch_size: int - patch大小，默认32
        sample_per_patch: int - 每个patch采样的点数，默认1
        point_label: int - 点的标签，1表示前景，0表示背景，默认1
    
    Return:
        points: (B, N, 2) - 点坐标，格式为(x, y)，即(列, 行)
        labels: (B, N) - 点标签，所有点的标签都是point_label
        注意：只从mask的前景区域（>0.5）采样，如果某个patch没有前景像素则跳过
        因此N <= num_patches * sample_per_patch，无效点用(-1, -1)填充，对应label为-1
    """
    # 处理mask维度
    if len(mask.shape) == 4:
        mask = mask.squeeze(1)  # (B, H, W)
    
    device = mask.device
    B, H, W = mask.shape
    
    # 计算patch网格大小
    num_patches_h = (H + patch_size - 1) // patch_size
    num_patches_w = (W + patch_size - 1) // patch_size
    
    # 初始化输出
    all_points = []
    all_labels = []
    
    for b in range(B):
        batch_points = []
        batch_labels = []
        
        for ph in range(num_patches_h):
            for pw in range(num_patches_w):
                # 计算当前patch的边界
                h_start = ph * patch_size
                h_end = min(h_start + patch_size, H)
                w_start = pw * patch_size
                w_end = min(w_start + patch_size, W)
                
                # 提取当前patch
                patch = mask[b, h_start:h_end, w_start:w_end]  # (patch_h, patch_w)
                
                # 在当前patch中从mask的前景区域随机采样点
                patch_h, patch_w = patch.shape
                
                for _ in range(sample_per_patch):
                    # 从mask的前景像素中采样（mask > 0.5的区域）
                    foreground_coords = torch.nonzero(patch > 0.5, as_tuple=False)  # [num_fg, 2]
                    
                    if foreground_coords.shape[0] > 0:
                        # 如果有前景像素，从中随机采样
                        idx = torch.randint(0, foreground_coords.shape[0], (1,), device=device).item()
                        local_y, local_x = foreground_coords[idx, 0].item(), foreground_coords[idx, 1].item()
                    else:
                        # 如果patch中没有前景像素，跳过该patch（不采样）
                        continue
                    
                    # 转换为全局坐标（注意：坐标格式是(x, y)，即(列, 行)）
                    global_x = w_start + local_x
                    global_y = h_start + local_y
                    
                    batch_points.append([global_x, global_y])
                    batch_labels.append(point_label)
        
        # 转换为tensor
        if len(batch_points) > 0:
            batch_points_tensor = torch.tensor(batch_points, device=device, dtype=torch.float32)  # (N, 2)
            batch_labels_tensor = torch.tensor(batch_labels, device=device, dtype=torch.float32)  # (N,)
        else:
            # 如果没有采样到任何点，创建空的tensor
            batch_points_tensor = torch.empty((0, 2), device=device, dtype=torch.float32)
            batch_labels_tensor = torch.empty((0,), device=device, dtype=torch.float32)
        
        all_points.append(batch_points_tensor)
        all_labels.append(batch_labels_tensor)
    
    # Stack成batch格式（注意：由于可能跳过某些patch，每个batch的点数可能不同）
    # 需要找到最大长度并padding
    max_points = max([p.shape[0] for p in all_points]) if all_points else 0
    
    if max_points > 0:
        # 对所有batch进行padding到相同长度
        padded_points = []
        padded_labels = []
        for points, labels in zip(all_points, all_labels):
            if points.shape[0] < max_points:
                # 用-1填充
                pad_points = torch.full((max_points - points.shape[0], 2), -1, device=device, dtype=points.dtype)
                pad_labels = torch.full((max_points - labels.shape[0],), -1, device=device, dtype=labels.dtype)
                points = torch.cat([points, pad_points], dim=0)
                labels = torch.cat([labels, pad_labels], dim=0)
            padded_points.append(points)
            padded_labels.append(labels)
        
        points = torch.stack(padded_points)  # (B, N, 2)
        labels = torch.stack(padded_labels)  # (B, N)
    else:
        # 如果所有batch都没有点，返回空tensor
        points = torch.empty((B, 0, 2), device=device, dtype=torch.float32)
        labels = torch.empty((B, 0), device=device, dtype=torch.float32)
    
    return points, labels