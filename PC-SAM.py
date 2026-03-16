import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import os
import time
from pathlib import Path
from data.dataloader import SamDataset, DataAugment
from torch.utils.data import DataLoader
from PIL import Image
from utils.util import show_img_mask, differentiable_opening, create_point_labels, get_randomized_prompt_parameters, show_tensor_image, generate_patch_labels_from_points, save_masks_with_points, sample_points_from_patches, draw_points, points_transform
from utils.tensor_process import DeNormalize
from utils.loss import DiceLoss, FocalLoss, SSLoss
from utils.lr_scheduler import LR_Scheduler
from utils.measure import compute_iou_binary, compute_F1
from utils.seed import seed_everything
from modeling.patch_constrained_sam import patch_constrained_sam
from collections import OrderedDict
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from utils.metric import SegmentationMetric
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="MaskPoint SAM Training/Validation with YAML config")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--valid', action='store_true', help='Run in validation mode')
    parser.add_argument('--valid_best', action='store_true', help='Use best model for validation')
    parser.add_argument('--valid_show_img_mask', action='store_true', help='Show images and predictions')
    parser.add_argument('--load_model', action='store_true', help='Load model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Override device (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--start_epoch', type=int, default=None, help='Override start epoch')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    parser.add_argument('--num_prompt_points', type=int, default=None, help='Override number of prompt points')
    parser.add_argument('--num_prompt_points_delta', type=float, default=None, help='Override number of prompt points delta')
    parser.add_argument('--fore_rate_delta', type=float, default=None, help='Override fore rate delta')
    parser.add_argument('--fore_rate', type=float, default=None, help='Override fore rate')
    parser.add_argument('--patch_size', type=int, default=None, help='Override patch size')
    parser.add_argument('--fn_op_kernel_size', type=int, default=None, help='Override patch size')
    parser.add_argument('--fp_op_kernel_size', type=int, default=None, help='Override patch size')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. 基本参数从yml读取，可被命令行覆盖
    base_num_prompt_points = args.num_prompt_points if args.num_prompt_points is not None else config.get('num_prompt_points', 50)
    base_fore_rate = args.fore_rate if args.fore_rate is not None else config.get('fore_rate', 0.7)
    prompt_points_delta = args.num_prompt_points_delta if args.num_prompt_points_delta is not None else config.get('num_prompt_points_delta', 0.5)
    fore_rate_delta = args.fore_rate_delta if args.fore_rate_delta is not None else config.get('fore_rate_delta', 0.5)
    patch_size = args.patch_size if args.patch_size is not None else config.get('patch_size', 32)
    fn_op_kernel_size = args.fn_op_kernel_size if args.fn_op_kernel_size is not None else config.get('fn_op_kernel_size', 3)
    fp_op_kernel_size = args.fp_op_kernel_size if args.fp_op_kernel_size is not None else config.get('fp_op_kernel_size', 13)
    fn_sample_points_per_patch = config.get('fn_sample_points_per_patch', 1)
    fp_sample_points_per_patch = config.get('fp_sample_points_per_patch', 1)
    model_name = config.get('model', "sam_vit_b_01ec64.pth")
    model_save_name = config.get('model_save_name')
    log_file = "logs/" + model_save_name + ".txt"
    Best_IoU = config.get('Best_IoU', 0.0)
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    start_epoch = args.start_epoch if args.start_epoch is not None else config.get('start_epoch', 0)
    epochs = args.epochs if args.epochs is not None else config.get('epochs', 200)
    lr = args.lr if args.lr is not None else config.get('lr', 1e-4)
    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 4)
    valid_mode = args.valid
    valid_best = args.valid_best
    valid_show_img_mask = args.valid_show_img_mask
    load_model = args.load_model or config.get('load_model', False)
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # model_save_name = model_save_name + ".pth"
    # 2. 路径参数

    input_dir = Path(config.get('input_dir', "input"))
    output_dir = Path(config.get('output_dir', "train-output"))
    gt_mask_dir = Path(config.get('gt_mask_dir', "gt-mask"))
    valid_input_dir = Path(config.get('valid_input_dir', "valid-input"))
    valid_output_dir = Path(config.get('valid_output_dir', "valid-output"))
    valid_mask_dir = Path(config.get('valid_mask_dir', "valid-mask"))
    checkpoint_dir = Path(config.get('checkpoint_dir', "sam-checkpoint"))
    model_checkpoint_save_dir = Path(config.get('model_checkpoint_save_dir', "train-checkpoint"))
    model_path = checkpoint_dir / model_name

    if valid_mode:
        input_dir = valid_input_dir
        output_dir = valid_output_dir
        gt_mask_dir = valid_mask_dir
        # batch_size = 1
    if valid_best:
        model_save_name = "best_" + model_save_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    # 3. 初始化模型
    model = patch_constrained_sam(
        sam_model=config.get('sam_model', "vit_b"),
        model_path=model_path,
        # original_size=tuple(config.get('original_size', (1024, 1024))),
    )
    model.to(device)
    # print(model)

    for param in model.parameters():
        param.requires_grad = False
        
    # LoRA injecting
    LORA_CONFIG = LoraConfig(
        r=8,                # LoRA 的秩（低秩分解维度）
        lora_alpha=32,      # 缩放因子
        target_modules=["attn.qkv", "attn.proj", "lin"],  # 想加 LoRA 的层
        lora_dropout=0.05,  # Dropout 比例
        bias="none",        # 是否对 bias 应用 LoRA
        )
    model.image_encoder = get_peft_model(model.image_encoder, LORA_CONFIG)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if load_model:
        checkpoint = torch.load(str(model_checkpoint_save_dir / model_save_name) + ".pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 4. 设置参数是否可训练
    # for param in model.parameters():
    #     param.requires_grad = False
    for param in model.prompt_decoder.parameters():
        param.requires_grad = True
    for param in model.mask_decoder.parameters():
        param.requires_grad = True
    for param in model.repair_patch_decoder.parameters():
        param.requires_grad = True
    for param in model.fusion_head.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e+6:.4f}M, Trainable parameters: {trainable_params/1e+6:.4f}M")

    # 获取需要更新的参数
    mask_decoder_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad and n.startswith("mask_decoder")
    ]
    print(f'mask_decoder parameters: {sum(p.numel() for p in mask_decoder_params)/1e+6:.4f}M')

    repair_patch_decoder_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad and n.startswith("repair_patch_decoder")
    ]
    print(f'repair_patch_decoder parameters: {sum(p.numel() for p in repair_patch_decoder_params)/1e+6:.4f}M')

    prompt_decoder_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad and n.startswith("prompt_decoder")
    ]
    print(f'prompt_decoder parameters: {sum(p.numel() for p in prompt_decoder_params)/1e+6:.4f}M')

    fusion_head_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad and n.startswith("fusion_head")
    ]
    print(f'fusion_head parameters: {sum(p.numel() for p in fusion_head_params)/1e+6:.4f}M')

    lora_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad 
        and not n.startswith("mask_decoder")
        and not n.startswith("repair_patch_decoder")
        and not n.startswith("prompt_decoder")
        and not n.startswith("fusion_head")
    ]
    print(f'LoRA trainable parameters: {sum(p.numel() for p in lora_params)/1e+6:.4f}M')

    # 优化器1:
    optimizer = torch.optim.AdamW([
        {"params": mask_decoder_params, "lr": lr},
        {"params": repair_patch_decoder_params, "lr": lr},
        {"params": prompt_decoder_params, "lr": lr},
        {"params": lora_params, "lr": lr * 10},
        {"params": fusion_head_params, "lr": lr * 10},
        ],
        weight_decay=5e-4
    )
    # 5. 数据集和dataloader
    dataset = SamDataset(image_dir=str(input_dir), gt_mask_dir=str(gt_mask_dir))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not valid_mode, num_workers=0)
    scheduler = LR_Scheduler(mode="poly", base_lr=lr, num_epochs=epochs, iters_per_epoch=len(dataloader), power=3, base_lr_groups=3)
    loss_dice = DiceLoss()
    loss_focal = FocalLoss()
    loss_ss = SSLoss(kernel_size=5, only_one=False)
    loss_mse = nn.MSELoss()
    Augmentor = DataAugment(brightness=0.2, contrast=0.4, saturation=0.4, device=device)

    # metric = SegmentationMetric(2)
    # repaired_part_metric = SegmentationMetric(2)
    # model_metric = SegmentationMetric(2)

    def check_nan_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"⚠️ NaN detected in output of {module.__class__.__name__}")
        elif isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    print(f"⚠️ NaN detected in output {i} of {module.__class__.__name__}")
        elif isinstance(output, dict) or isinstance(output, OrderedDict):
            for k, out in output.items():
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    print(f"⚠️ NaN detected in output[{k}] of {module.__class__.__name__}")

    # for name, layer in model.named_modules():
    #     layer.register_forward_hook(check_nan_hook)

    # 6. 训练/验证流程
    if not valid_mode:
        model.train()
        for epoch in range(start_epoch, epochs):
            seed_everything(seed + epoch)

            Total_Loss = 0.0
            Prompt_IoU = 0.0
            IoU = 0.0
            Repaired_Part_IoU = 0.0
            Prompt_IoU_Predict = 0.0
            Prompt_IoU_Count = 0
            IoU_Count = 0
            Repaired_Part_Count = 0
            Model_IoU = 0.0
            Model_IoU_Count = 0
            Model_TwoMask_IoU = 0.0
            Model_TwoMask_IoU_Count = 0

            dataloader_iter = iter(dataloader)
            tqdm_loader = tqdm(dataloader_iter)
            for idx, data in enumerate(tqdm_loader):
                # for name, param in model.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print("NaN in grad:", name)

                input_image = data['image'].to(device)
                input_image_name = data['image_name']
                original_size = data['original_size'].cpu().numpy()
                gt_mask = data['gt_mask'].to(device)
                not_white_mask = data['not_white_mask'].to(device)
                
                input_image, gt_mask, not_white_mask = Augmentor.train_process(input_image, gt_mask, not_white_mask)

                gt_mask = not_white_mask * gt_mask

                num_prompt_points, fore_rate = get_randomized_prompt_parameters(base_num_prompt_points, base_fore_rate, prompt_points_delta, fore_rate_delta)
                original_fore_points, original_back_points = create_point_labels(gt_mask, num_points=num_prompt_points, fore_rate=fore_rate)
                pos_label, neg_label = generate_patch_labels_from_points(gt_mask, original_fore_points, original_back_points, patch_size=patch_size)
                delete_label = gt_mask - neg_label.unsqueeze(1)
                model_label = torch.max(pos_label, neg_label)

                # show_tensor_image(pos_label.unsqueeze(1), is_mask=True)
                # show_tensor_image(neg_label.unsqueeze(1), is_mask=True)
                # show_tensor_image(model_label.unsqueeze(1), is_mask=True)
                # print(not_white_mask)
                # print(not_white_mask.shape)
                # print("img shape: ", input_image.shape)
                # print("mask shape: ", gt_mask.shape)

                # if torch.sum(not_white_mask) == 0:
                #     print("All white!")
                # show_tensor_image(input_image, input_image_name, is_mask=False)
                # show_tensor_image(gt_mask, "gt_mask", is_mask=True)
                # show_tensor_image(not_white_mask, "not_white_mask", is_mask=True)
                # input("Paused for visualization. Press Enter to continue...")

                fore_points = points_transform(original_fore_points, original_size[0], 1024)
                back_points = points_transform(original_back_points, original_size[0], 1024)

                optimizer.zero_grad()

                # with autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                (upscaled_masks, 
                 repaired_masks,
                 model_logits,
                 prompt_masks,
                 iou_prompt_predict) = model(input_image, 
                                            fore_points,
                                            back_points,
                                            original_size[0])

                # model_logits = upscaled_masks + repaired_masks
                # pred_prob = torch.sigmoid(upscaled_masks)   # B, 1, H, W
                # repaired_prob = torch.sigmoid(repaired_masks)
                # model_prob = torch.sigmoid(model_logits)

                # 取 pred_mask 和 hard_part_mask 在原始图像非全白区域的部分
                # pred_mask_nonwhite_keepdim = pred_mask * not_white_mask
                # hard_part_mask_nonwhite_keepdim = hard_part_mask * not_white_mask
                # gt_mask_nonwhite_keepdim = gt_mask * not_white_mask
                
                with torch.no_grad():
                    prompt_bin = (prompt_masks > 0).float().squeeze(1)
                    mask_bin = (upscaled_masks > 0).float().squeeze(1)     # B, H, W
                    repaired_part_mask_bin = (repaired_masks > 0).float().squeeze(1)
                    model_mask = (model_logits > 0).float()
                    model_twomask = ((mask_bin + repaired_part_mask_bin) > 0).float()

                    prompt_iou = compute_iou_binary(prompt_bin, gt_mask, mean=False, eps=True)
                    iou = compute_iou_binary(mask_bin, neg_label, eps=True)
                    repaired_part_iou = compute_iou_binary(repaired_part_mask_bin, pos_label, eps=True)
                    model_iou = compute_iou_binary(model_mask, model_label, eps=True)
                    model_twomask_iou = compute_iou_binary(model_twomask, model_label, eps=True)

                # delete_part = upscaled_masks * delete_label

                # show_tensor_image((1-neg_label).unsqueeze(1)*gt_mask, is_mask=True)
                # show_tensor_image(torch.sigmoid(-upscaled_masks) * gt_mask, is_mask=False)
                # show_tensor_image(delete_part, is_mask=False)
                # show_tensor_image(delete_label, is_mask=True)

                # loss_delete_part = 0.3 * loss_dice(upscaled_masks, neg_label, gt_mask) + 0.7 * loss_focal(upscaled_masks, neg_label, gt_mask)
                loss_iou_prompt = loss_mse(iou_prompt_predict, prompt_iou)
                loss_prompt = 0.3 * loss_dice(prompt_masks, gt_mask) + + 0.65 * loss_focal(prompt_masks, gt_mask) + 0.05 * loss_ss(prompt_masks, gt_mask)
                loss_neg = 0.3 * loss_dice(upscaled_masks, neg_label) + 0.7 * loss_focal(upscaled_masks, neg_label)
                loss_pos = 0.3 * loss_dice(repaired_masks, pos_label) + 0.7 * loss_focal(repaired_masks, pos_label)
                loss_model = 0.3 * loss_dice(model_logits, model_label) + 0.7 * loss_focal(model_logits, model_label)
                loss = loss_neg + loss_pos + loss_model + loss_prompt + loss_iou_prompt #+ loss_delete_part
                # loss = loss_prompt + loss_iou_prompt
                # print(loss_delete_part)
                scheduler(optimizer, idx, epoch, Best_IoU)

                loss.backward()
                optimizer.step()

                Total_Loss += loss.item()
                avg_loss = Total_Loss / (idx+1)

                iou_prompt_predict = torch.mean(iou_prompt_predict, dim=0)
                prompt_iou = torch.mean(prompt_iou, dim=0)

                Prompt_IoU_Predict += iou_prompt_predict.item()
                avg_iou_prompt_predict = Prompt_IoU_Predict / (idx+1)

                if not torch.isnan(prompt_iou):
                    Prompt_IoU += prompt_iou
                    Prompt_IoU_Count += 1
                if not torch.isnan(iou):
                    IoU += iou
                    IoU_Count += 1
                if not torch.isnan(repaired_part_iou):
                    Repaired_Part_IoU += repaired_part_iou
                    Repaired_Part_Count += 1
                if not torch.isnan(model_iou):
                    Model_IoU += model_iou
                    Model_IoU_Count += 1
                if not torch.isnan(model_twomask_iou):
                    Model_TwoMask_IoU += model_twomask_iou
                    Model_TwoMask_IoU_Count += 1

                if Prompt_IoU_Count != 0:
                    avg_prompt_iou = Prompt_IoU / Prompt_IoU_Count
                else:
                    avg_prompt_iou = 0.0
                if IoU_Count != 0:
                    avg_iou = IoU / IoU_Count
                else:
                    avg_iou = 0.0
                if Repaired_Part_Count != 0:
                    avg_repaired_part_iou = Repaired_Part_IoU / Repaired_Part_Count
                else:
                    avg_repaired_part_iou = 0.0
                if Model_IoU_Count != 0:
                    avg_model_iou = Model_IoU / Model_IoU_Count
                else:
                    avg_model_iou = 0.0
                if Model_TwoMask_IoU_Count != 0:
                    avg_twomask_iou = Model_TwoMask_IoU / Model_TwoMask_IoU_Count
                else:
                    avg_twomask_iou = 0.0

                # stage1_hard_part_label = differentiable_opening(stage1_hard_part_label, erosion_kernel_size=5, dilation_kernel_size=3)
                # stage1_hard_road_part_logits = differentiable_opening(stage1_hard_road_part_logits, erosion_kernel_size=5, dilation_kernel_size=3)
                # stage1_hard_part_mask_bin = differentiable_opening(stage1_hard_part_mask_bin.unsqueeze(1), erosion_kernel_size=5, dilation_kernel_size=3)
                
                # print(iou)
                # show_tensor_image(input_image, input_image_name, is_mask=False)
                # show_tensor_image(neg_label.unsqueeze(1), "neg_label", is_mask=True)
                # show_tensor_image(pos_label.unsqueeze(1), "pos_label", is_mask=True)
                # show_tensor_image(not_white_mask, "not_white_mask", is_mask=True)
                # input("Paused for visualization. Press Enter to continue...")

                if idx % 40 == 0:
                    print(f"\nEpoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(dataloader)}],") 
                    print(f"Loss: {avg_loss:.4f}, Neg IoU: {avg_iou:.4f}, Additional Part IoU: {avg_repaired_part_iou:.4f}, Model Seg Head IoU: {avg_model_iou:.4f}, Model Two Mask IoU: {avg_twomask_iou:.4f}, Prompt Mask IoU: {avg_prompt_iou:.4f}, Prompt Mask IoU Predict: {avg_iou_prompt_predict:.4f}")

                # save_masks_with_points(
                #     input_image = DeNormalize(input_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #     gt_mask = gt_mask,
                #     mask_bin=mask_bin,
                #     repaired_part_mask_bin=repaired_part_mask_bin,
                #     fore_points=original_fore_points,
                #     back_points=original_back_points,
                #     output_dir=output_dir,
                #     input_image_name=input_image_name,
                #     draw_points_on_mask=True
                # )

            if avg_iou > Best_IoU:
                Best_IoU = avg_iou
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                    str(model_checkpoint_save_dir / ("best_" + model_save_name + ".pth")))

            if not os.path.exists(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"{model_save_name} Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Neg IoU: {avg_iou:.4f}, Additional Part IoU: {avg_repaired_part_iou:.4f}, Model IoU:{avg_model_iou:.4f}, Model Two Mask IoU: {avg_twomask_iou:.4f}, Prompt IoU: {avg_prompt_iou:.4f}, IoU Prompt Predict: {avg_iou_prompt_predict:.4f}\n")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },
                str(model_checkpoint_save_dir / (model_save_name + ".pth")))

    else:
        with torch.no_grad():
            model.eval()

            test_iou_log = open('logs/'+model_save_name+'_test_iou.log','w')
            print("****************", file=test_iou_log)
            test_iou_log = open('logs/'+model_save_name+'_test_iou.log','a')
            print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", file=test_iou_log)

            Total_Loss = 0.0
            Loss_Count = 0
            Prompt_IoU = 0.0
            IoU = 0.0
            Repaired_Part_IoU = 0.0
            Prompt_IoU_Predict = 0.0
            F1_Fusion = 0.0
            F1_TwoMask = 0.0
            Prompt_IoU_Count = 0
            IoU_Count = 0
            Repaired_Part_Count = 0
            Model_IoU = 0.0
            Model_IoU_Count = 0
            Model_TwoMask_IoU = 0.0
            Model_TwoMask_IoU_Count = 0
            F1_Fusion_Count = 0
            F1_TwoMask_Count = 0

            dataloader_iter = iter(dataloader)
            tqdm_loader = tqdm(dataloader_iter)
            for idx, data in enumerate(tqdm_loader):

                origin_input_image = data['image'].to(device)
                input_image_name = data['image_name']
                original_size = data['original_size'].cpu().numpy()
                gt_mask = data['gt_mask'].to(device)
                # print(gt_mask.shape)
                input_image, gt_mask = Augmentor.val_process(origin_input_image, gt_mask)

                # Stage 1
                image_embedding = model.get_image_embedding(input_image)
                # print(image_embedding.shape)
                auto_upscaled_masks, hr_masks, iou_prompt_predict = model.auto_segment(image_embedding, 
                                                                                           original_size[0])
                auto_mask_bin = (auto_upscaled_masks > 0).float().detach()
                hr_bin = (torch.sigmoid(hr_masks) > 0.2).float()
                # prompt_diff = differentiable_opening(prompt_bin, erosion_kernel_size=3, dilation_kernel_size=3)
                
                fp_mask = auto_mask_bin * (1 - gt_mask)
                fp_mask = differentiable_opening(fp_mask, erosion_kernel_size=fp_op_kernel_size, dilation_kernel_size=3)
                original_back_points = sample_points_from_patches(fp_mask, patch_size=patch_size, sample_per_patch=fp_sample_points_per_patch, point_label=0)
                back_points = points_transform(original_back_points, original_size[0], 1024)

                # Stage 2 neg segment
                # back_points_p = torch.empty((1, 0, 2)).to(device)
                # back_points_l = torch.empty((1, 0)).to(device)
                # back_points = (back_points_p, back_points_l)

                upscaled_masks = model.neg_segment(image_embedding, back_points, original_size[0])
                neg_prob = torch.sigmoid(upscaled_masks)   # B, 1, H, W
                neg_mask_bin = (neg_prob > 0.5).float().detach()

                fn_mask = (1 - neg_mask_bin) * gt_mask
                fn_mask = differentiable_opening(fn_mask, erosion_kernel_size=fn_op_kernel_size, dilation_kernel_size=3)
                original_fore_points = sample_points_from_patches(fn_mask, patch_size=patch_size, sample_per_patch=fn_sample_points_per_patch, point_label=1)
                fore_points = points_transform(original_fore_points, original_size[0], 1024)
                pos_label, _ = generate_patch_labels_from_points(gt_mask, original_fore_points, original_back_points, patch_size=patch_size)
                
                # Stage 3 pos segment
                repaired_masks = model.pos_segment(image_embedding, fore_points, original_size[0])
                repaired_prob = torch.sigmoid(repaired_masks)
                repaired_part_mask_bin = (repaired_prob > 0.5).float().detach()

                # Mask fusion
                model_logits = model.mask_fusion(upscaled_masks, repaired_masks)
                model_fusionmask = (model_logits > 0).float().detach()

                # upscaled_masks, repaired_masks, model_logits = model.prompt_segment(image_embedding,
                #                                                                     fore_points,
                #                                                                     back_points,
                #                                                                     original_size[0],
                #                                                                     )
                # model_logits = upscaled_masks + repaired_masks

                model_twomask = ((neg_mask_bin + repaired_part_mask_bin) > 0).float()

                prompt_iou = compute_iou_binary(hr_bin, gt_mask)
                iou = compute_iou_binary(neg_mask_bin, gt_mask)
                repaired_part_iou = compute_iou_binary(repaired_part_mask_bin, pos_label)
                model_iou = compute_iou_binary(model_fusionmask, gt_mask)
                model_twomask_iou = compute_iou_binary(model_twomask, gt_mask)
                f1_score_fusion = compute_F1(model_fusionmask, gt_mask)
                f1_score_twomask = compute_F1(model_twomask, gt_mask)

                # metric.CM(gt_mask.cpu(), mask_bin.squeeze(1).cpu().data.numpy())
                # repaired_part_metric.CM(pos_label.cpu(), repaired_part_mask_bin.squeeze(1).cpu().data.numpy())
                # model_metric.CM(gt_mask.cpu(), model_fusionmask.squeeze(1).cpu().data.numpy())

                loss_iou_prompt = loss_mse(iou_prompt_predict, prompt_iou.unsqueeze(0).reshape(1, 1))
                loss_prompt = 0.3 * loss_dice(hr_masks, gt_mask) + 0.7 * loss_ss(hr_masks, gt_mask)
                loss_neg = 0.3 * loss_dice(upscaled_masks, gt_mask) + 0.7 * loss_focal(upscaled_masks, gt_mask)
                loss_pos = 0.3 * loss_dice(repaired_masks, pos_label) + 0.7 * loss_focal(repaired_masks, pos_label)# + 0.05 * loss_ss(repaired_prob, pos_label)
                loss_model = 0.3 * loss_dice(model_logits, gt_mask) + 0.7 * loss_focal(model_logits, gt_mask)
                # loss = loss_neg + loss_pos + loss_model + loss_prompt + loss_iou_prompt
                loss = loss_prompt + loss_iou_prompt

                Prompt_IoU_Predict += iou_prompt_predict.item()
                avg_iou_prompt_predict = Prompt_IoU_Predict / (idx+1)

                if not torch.isnan(loss):
                    Total_Loss += loss.item()
                    Loss_Count += 1
                avg_loss = Total_Loss / (idx+1)

                if not torch.isnan(prompt_iou):
                    Prompt_IoU += prompt_iou
                    Prompt_IoU_Count += 1
                if not torch.isnan(iou):
                    IoU += iou
                    IoU_Count += 1
                if not torch.isnan(repaired_part_iou):
                    Repaired_Part_IoU += repaired_part_iou
                    Repaired_Part_Count += 1
                if not torch.isnan(model_iou):
                    Model_IoU += model_iou
                    Model_IoU_Count += 1
                if not torch.isnan(model_twomask_iou):
                    Model_TwoMask_IoU += model_twomask_iou
                    Model_TwoMask_IoU_Count += 1
                if not torch.isnan(f1_score_fusion):
                    F1_Fusion += f1_score_fusion
                    F1_Fusion_Count += 1
                if not torch.isnan(f1_score_twomask):
                    F1_TwoMask += f1_score_twomask
                    F1_TwoMask_Count += 1

                avg_prompt_iou = Prompt_IoU / Prompt_IoU_Count
                avg_iou = IoU / IoU_Count
                avg_repaired_part_iou = Repaired_Part_IoU / Repaired_Part_Count
                avg_model_iou = Model_IoU / Model_IoU_Count
                avg_twomask_iou = Model_TwoMask_IoU / Model_TwoMask_IoU_Count
                avg_f1_fusion = F1_Fusion / F1_Fusion_Count
                avg_f1_twomask = F1_TwoMask / F1_TwoMask_Count
                
                tqdm_loader.set_description(
                    f"Step [{idx+1}/{len(dataloader)}]"
                )
                tqdm_loader.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "Auto Segment IoU": f"{avg_iou:.4f}",
                    "Additional Part IoU": f"{avg_repaired_part_iou:.4f}",
                    "Model Fusion IoU": f"{avg_model_iou:.4f}",
                    "Model Two Mask IoU": f"{avg_twomask_iou:.4f}",
                    "Model Fusion F1": f"{avg_f1_fusion:.4f}",
                    "Model Two Mask F1": f"{avg_f1_twomask:.4f}",
                    "Prompt IoU": f"{avg_prompt_iou:.4f}",
                    "IoU Prompt Predict": f"{avg_iou_prompt_predict:.4f}",
                })
                
                if idx % 100 == 0:
                    print(f"\nStep [{idx+1}/{len(dataloader)}],") 
                    print(f"Loss: {avg_loss:.4f}, Auto Segment IoU: {avg_iou:.4f}, Additional Part IoU: {avg_repaired_part_iou:.4f}, Model Fusion IoU: {avg_model_iou:.4f}, Model Two Mask IoU: {avg_twomask_iou:.4f}, Model Fusion F1: {avg_f1_fusion:.4f}, Model Two Mask F1: {avg_f1_twomask:.4f}, Prompt IoU: {avg_prompt_iou:.4f}, IoU Prompt Predict: {avg_iou_prompt_predict:.4f}")
                
                # for i in range(input_image.shape[0]):
                #     im_img = Image.fromarray(origin_input_image[i].cpu().numpy())
                #     im_img.save(output_dir / f"{input_image_name[i].replace('.jpg', '_0_img.png')}")
                #     im_gt = Image.fromarray(gt_mask[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255)
                #     im_gt.save(output_dir / f"{input_image_name[i].replace('.jpg', '_1_gt.png')}")
                #     auto_mask_img = auto_mask_bin[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     im_auto_mask = Image.fromarray(auto_mask_img)
                #     im_auto_mask.save(output_dir / f"{input_image_name[i].replace('.jpg', '_2_automask.png')}")
                #     mask_img = neg_mask_bin[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     mask_img = draw_points(mask_img, original_back_points[0][i], color=(255, 0, 0), point_size=13)
                #     im_mask = Image.fromarray(mask_img)
                #     im_mask.save(output_dir / f"{input_image_name[i].replace('.jpg', '_3_negmask.png')}")
                #     repaired_img = repaired_part_mask_bin[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     repaired_img = draw_points(repaired_img, original_fore_points[0][i], color=(0, 255, 0), point_size=13)
                #     im_repaired = Image.fromarray(repaired_img)
                #     im_repaired.save(output_dir / f"{input_image_name[i].replace('.jpg', '_4_posmask.png')}")
                #     model_twomask_img = model_twomask[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     im_model_twomask = Image.fromarray(model_twomask_img)
                #     im_model_twomask.save(output_dir / f"{input_image_name[i].replace('.jpg', '_5_twomask.png')}")
                #     model_FusionHead_mask = model_fusionmask[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     im_model_FusionHead_mask = Image.fromarray(model_FusionHead_mask)
                #     im_model_FusionHead_mask.save(output_dir / f"{input_image_name[i].replace('.jpg', '_6_fusionmask.png')}")
                #     prompt_mask_img = prompt_bin[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                #     im_prompt_mask = Image.fromarray(prompt_mask_img)
                #     im_prompt_mask.save(output_dir / f"{input_image_name[i].replace('.jpg', '_7_promptmask.png')}")
                
                print(f'{input_image_name} Auto Segment IoU: {iou.item():.4f}, Additional Part IoU: {repaired_part_iou.item():.4f}, Model Fusion IoU: {model_iou.item():.4f}, Model Two Mask IoU: {model_twomask_iou.item():.4f}, Model Fusion F1: {f1_score_fusion.item():.4f}, Model Two Mask F1: {f1_score_twomask.item():.4f}, Prompt IoU: {prompt_iou.item():.4f}, IoU Prompt Predict: {iou_prompt_predict.item():.4f}', file=test_iou_log)
                
                if valid_show_img_mask:
                    print(f"image name: {input_image_name}, Auto Segment IoU: {iou.item():.4f}, Additional Part IoU: {repaired_part_iou.item():.4f}, Model Fusion IoU: {model_iou.item():.4f}, Model Two Mask IoU: {model_twomask_iou.item():.4f}, Model Fusion F1: {f1_score_fusion.item():.4f}, Model Two Mask F1: {f1_score_twomask.item():.4f}, Prompt IoU: {prompt_iou.item():.4f}, IoU Prompt Predict: {iou_prompt_predict.item():.4f}")
                    show_img_mask(
                        img=input_image.squeeze(0).cpu(),
                        gt_mask=gt_mask.squeeze(0).cpu(),
                        logits=upscaled_masks.squeeze(0).detach().cpu(),
                        repaired_part_logits=repaired_masks.squeeze(0).detach().cpu(),
                        prompt_logits=hr_masks.squeeze(0).detach().cpu(),
                        fore_points = original_fore_points[0].squeeze(0).cpu(),
                        back_points = original_back_points[0].squeeze(0).cpu(),
                        threshold=0,
                    )
                    plt.show()
                    input("")

            # oare_iou = metric.IOU()[1]
            # oare_repaired_part_iou = repaired_part_metric.IOU()[1]
            # oare_model_iou = model_metric.IOU()[1]

            valid_log = open('logs/'+model_save_name+'_test.log','a')
            print(f'Valid completed.')
            print("****************", file=valid_log)
            print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", file=valid_log)
            print(f"batch size: {batch_size}", file=valid_log)
            print(f"Loss: {avg_loss:.4f}", file=valid_log)
            print(f"fn differentiable opening erosion_kernel size: {fn_op_kernel_size}", file=valid_log)
            print(f"fp differentiable opening erosion_kernel size: {fp_op_kernel_size}", file=valid_log)
            print(f'fn sample_points_per_patch: {fn_sample_points_per_patch}', file=valid_log)
            print(f'fp sample_points_per_patch: {fp_sample_points_per_patch}', file=valid_log)
            print(f"Auto Segment IoU: {avg_iou.item():.4f}", file=valid_log)
            print(f"Additional Part IoU: {avg_repaired_part_iou.item():.4f}", file=valid_log)
            print(f"Model Fusion Head IoU: {avg_model_iou.item():.4f}, Model Two Mask IoU: {avg_twomask_iou.item():.4f}", file=valid_log)
            print(f"Model Fusion F1: {avg_f1_fusion.item():.4f}, Model Two Mask F1: {avg_f1_twomask.item():.4f}", file=valid_log)
            print(f"Prompt IoU: {avg_prompt_iou.item():.4f}", file=valid_log)
            print(f"IoU Prompt Predict: {avg_iou_prompt_predict:.4f}", file=valid_log)

if __name__ == "__main__":
    main()
