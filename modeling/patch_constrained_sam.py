import torch
import torch.nn as nn
import copy
from segmentanything.segment_anything.build_sam import sam_model_registry

class patch_constrained_sam(nn.Module):
    def __init__(self, sam_model, model_path):
        super().__init__()

        sam = sam_model_registry[sam_model](checkpoint=str(model_path))
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.repair_patch_decoder = copy.deepcopy(self.mask_decoder)
        self.prompt_decoder = copy.deepcopy(self.mask_decoder)
        self.postprocess_masks = sam.postprocess_masks
        # self.seg_head = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.fusion_head = MaskFusion()

        self.input_size = None

    def forward(self, x, fore_points=None, back_points=None, original_size=None):

        im_outputs = self.image_encoder(x)
        image_embedding = im_outputs[-1]

        # Forward pass
        back_sparse_embeddings, back_dense_embeddings = self.prompt_encoder(
            points=back_points,
            boxes=None,
            masks=None,
        )
        # Decode the masks
        low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=back_sparse_embeddings,
                dense_prompt_embeddings=back_dense_embeddings,
                multimask_output=False,
            )
        
        # Patch Repair
        fore_sparse_embeddings, fore_dense_embeddings = self.prompt_encoder(
            points=fore_points,
            boxes=None,
            masks=None,
        )
        patch_repair_low_res_masks, _ = self.repair_patch_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=fore_sparse_embeddings,
                dense_prompt_embeddings=fore_dense_embeddings,
                multimask_output=False,
            )
        
        # Generate prompt masks
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        prompt_low_res_masks, iou_prompt = self.prompt_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        
        # PostProcess
        upscaled_masks = self.postprocess_masks(
            low_res_masks,
            input_size=x.shape[-2:],
            original_size=(original_size[0], original_size[1])
        )
        repaired_masks = self.postprocess_masks(
            patch_repair_low_res_masks,
            input_size=x.shape[-2:],
            original_size=(original_size[0], original_size[1])
        )
        prompt_masks = self.postprocess_masks(
            prompt_low_res_masks,
            input_size=x.shape[-2:],
            original_size=(original_size[0], original_size[1])
        )
        # seg_logits = self.seg_head(torch.cat([upscaled_masks, self.relu(repaired_masks)], dim=1))
        seg_logits = self.fusion_head(upscaled_masks, repaired_masks)

        return upscaled_masks, repaired_masks, seg_logits, prompt_masks, iou_prompt

    def get_image_embedding(self, x):
        im_outputs = self.image_encoder(x)
        image_embedding = im_outputs[-1]
        self.input_size = x.shape[-2:]

        return image_embedding

    def auto_segment(self, image_embedding, original_size=None):

        # im_outputs = self.image_encoder(x)
        # image_embedding = im_outputs[-1]

        # Forward pass

        none_points_p = torch.empty((1, 0, 2)).to(image_embedding.device)
        none_points_l = torch.empty((1, 0)).to(image_embedding.device)
        none_points = (none_points_p, none_points_l)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=none_points,
            boxes=None,
            masks=None,
        )
        # Decode the masks
        low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        # Generate prompt masks
        prompt_low_res_masks, iou_prompt = self.prompt_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )    
        # PostProcess
        upscaled_masks = self.postprocess_masks(
            low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )
        prompt_masks = self.postprocess_masks(
            prompt_low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )

        return upscaled_masks, prompt_masks, iou_prompt
    
    def prompt_segment(self, image_embedding, fore_points=None, back_points=None, original_size=None):

        # Forward pass
        back_sparse_embeddings, back_dense_embeddings = self.prompt_encoder(
            points=back_points,
            boxes=None,
            masks=None,
        )
        # Decode the masks
        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=back_sparse_embeddings,
                dense_prompt_embeddings=back_dense_embeddings,
                multimask_output=False,
            )
        
        # Patch Repair
        fore_sparse_embeddings, fore_dense_embeddings = self.prompt_encoder(
            points=fore_points,
            boxes=None,
            masks=None,
        )
        patch_repair_low_res_masks, _ = self.repair_patch_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=fore_sparse_embeddings,
                dense_prompt_embeddings=fore_dense_embeddings,
                multimask_output=False,
            )

        # PostProcess
        upscaled_masks = self.postprocess_masks(
            low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )
        repaired_masks = self.postprocess_masks(
            patch_repair_low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )
        seg_logits = self.fusion_head(upscaled_masks, repaired_masks)

        return upscaled_masks, repaired_masks, seg_logits

    def neg_segment(self, image_embedding, back_points=None, original_size=None):
        # Forward pass
        back_sparse_embeddings, back_dense_embeddings = self.prompt_encoder(
            points=back_points,
            boxes=None,
            masks=None,
        )
        # Decode the masks
        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=back_sparse_embeddings,
                dense_prompt_embeddings=back_dense_embeddings,
                multimask_output=False,
            )
        # PostProcess
        upscaled_masks = self.postprocess_masks(
            low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )
        return upscaled_masks

    def pos_segment(self, image_embedding, fore_points=None, original_size=None):
        # Patch Repair
        fore_sparse_embeddings, fore_dense_embeddings = self.prompt_encoder(
            points=fore_points,
            boxes=None,
            masks=None,
        )
        patch_repair_low_res_masks, _ = self.repair_patch_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=fore_sparse_embeddings,
                dense_prompt_embeddings=fore_dense_embeddings,
                multimask_output=False,
            )
        # PostProcess
        repaired_masks = self.postprocess_masks(
            patch_repair_low_res_masks,
            input_size=self.input_size,
            original_size=(original_size[0], original_size[1])
        )

        return repaired_masks

    def mask_fusion(self, upscaled_masks, repaired_masks):
        fusion_logits = self.fusion_head(upscaled_masks, repaired_masks)
        
        return fusion_logits

class MaskFusion(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.mask1_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.mask1_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mask1_minpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mask2_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.mask2_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mask2_minpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.SpatialAttn1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
        )
        self.SpatialAttn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, padding=2),
            nn.BatchNorm2d(4),
            nn.GELU(),
        )
        self.SpatialAttn3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            nn.BatchNorm2d(4),
            nn.GELU(),
        )
        self.Attn = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2):

        x1_bn = self.bn1(x1)
        x2_bn = self.bn2(x2)

        x1_avg = self.mask1_avgpool(x1_bn)
        x1_max = self.mask1_maxpool(x1_bn)
        x1_min = -self.mask1_minpool(-x1_bn)
        x1_f = torch.cat([x1_bn, x1_avg, x1_max, x1_min], dim=1)

        x2_avg = self.mask2_avgpool(x2_bn)
        x2_max = self.mask2_maxpool(x2_bn)
        x2_min = -self.mask2_minpool(-x2_bn)
        x2_f = torch.cat([x2_bn, x2_avg, x2_max, x2_min], dim=1)

        x = torch.cat([x1_f, x2_f], dim=1)
        x_attn1 = self.SpatialAttn1(x)
        x_attn2 = self.SpatialAttn2(x)
        x_attn3 = self.SpatialAttn3(x)
        x_attn = self.Attn(torch.cat([x_attn1, x_attn2, x_attn3], dim=1))

        attn_score = self.sigmoid(x_attn)

        output = x1 * attn_score + x2 * (1 - attn_score)

        return output