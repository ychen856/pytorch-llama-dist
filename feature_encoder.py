import torch
import torch.nn as nn
import torch.nn.functional as F
class FlexibleTopKEncoder(nn.Module):
    def __init__(self, vocab_dim=32000, max_bottleneck_dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(vocab_dim)
        self.shared_proj = nn.Sequential(
            nn.Linear(vocab_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024)
        )
        self.bottleneck_heads = nn.ModuleDict({
            str(d): nn.Linear(1024, d) for d in [64, 128, 256, 512, 768]
        })

    def forward(self, topk_logits, bottleneck_dim):
        x = self.norm(topk_logits)
        h = self.shared_proj(x)
        z = self.bottleneck_heads[str(bottleneck_dim)](h)
        return z
