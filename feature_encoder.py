import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleTopKEncoder(nn.Module):
    def __init__(self, input_dim=4096, max_bottleneck_dim=512):
        super().__init__()

        # 定義所有 bottleneck 對應的 head
        self.token_encoder_heads = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, d)
            ) for d in [64, 128, 256, 512, 768]
        })

        # optional normalization for stability
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, bottleneck_dim):
        """
        x: [B, k, input_dim]  → top-k token logits
        bottleneck_dim: int
        """
        x = self.norm(x)  # normalize before projecting (optional but helps)
        encoder = self.token_encoder_heads[str(bottleneck_dim)]
        z = encoder(x)  # [B, k, bottleneck_dim]
        return z