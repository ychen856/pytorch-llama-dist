import torch
import torch.nn as nn

class FlexibleDecoder(nn.Module):
    def __init__(self, hidden_dim=4096, seq_len=1024, max_bottleneck_dim=512):
        super().__init__()
        self.seq_len = seq_len
        self.default_token = nn.Parameter(torch.zeros(hidden_dim))

        self.token_decoder_heads = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(d, 1024),
                nn.ReLU(),
                nn.Linear(1024, hidden_dim)
            ) for d in [64, 128, 256, 512]
        })

    def forward(self, z, topk_idx, bottleneck_dim):
        """
        z: [B, k, bottleneck_dim]
        topk_idx: [B, k]
        bottleneck_dim: int
        """
        B, k, _ = z.shape
        device = z.device
        decoder = self.token_decoder_heads[str(bottleneck_dim)]
        decoded_topk = decoder(z)  # [B, k, hidden_dim]

        # Build full feature map
        full_feat = self.default_token.expand(B, self.seq_len, -1).clone()
        for b in range(B):
            full_feat[b, topk_idx[b]] = decoded_topk[b]
        return full_feat  # → to Layer k+1