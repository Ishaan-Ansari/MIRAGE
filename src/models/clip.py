"""CLIP module that converts images to embeddings."""

import clip
from torch import nn

import configs.config as CFG


class vision_encoder(nn.Module):
    """CLIP Vision Encoder adapted for medical image analysis."""

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=True,
        trainable=False,
    ):
        """Initialize the CLIP Vision Encoder."""
        super().__init__()
        self.model, _ = clip.load(model_name, device="cpu", jit=False)
        for p in self.model.visual.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """Forward pass through the vision encoder."""
        return self.model.encode_image(x)
