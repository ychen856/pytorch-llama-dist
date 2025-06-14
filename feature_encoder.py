import torch
import torch.nn as nn

class FlexibleTopKEncoder(nn.Module):
    def __init__(self, vocab_dim=32000, max_bottleneck_dim=512):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(vocab_dim, 1024),
            nn.ReLU()
        )
        self.bottleneck_heads = nn.ModuleDict({
            str(d): nn.Linear(1024, d) for d in [64, 128, 256, 512, 768]
        })

    def forward(self, topk_logits, bottleneck_dim):
        """
        topk_logits: [B, k, vocab_dim]
        bottleneck_dim: int, current bottleneck size
        """
        h = self.shared_proj(topk_logits)  # [B, k, 1024]
        z = self.bottleneck_heads[str(bottleneck_dim)](h)  # [B, k, bottleneck_dim]
        return z