import torch
import torch.nn as nn
from segmentanything.segment_anything.build_sam import sam_model_registry


class sam(nn.Module):
    def __init__(self, sam_model, model_path, original_size):
        super().__init__()

        sam = sam_model_registry[sam_model](checkpoint=str(model_path))

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.postprocess_masks = sam.postprocess_masks
        self.original_size = original_size

    def forward(self, x, points=None, masks=None):
        # Generate prompts mask
        # Forward pass

        image_embedding = self.image_encoder(x)[-1]
        # print(image_embedding.shape)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # Decode the masks
        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        # PostProcess
        upscaled_masks = self.postprocess_masks(
            low_res_masks,
            input_size=x.shape[-2:],
            original_size=self.original_size[0]
        )

        return upscaled_masks, iou_predictions, low_res_masks
