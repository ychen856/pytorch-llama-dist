
import torch
import torch.nn as nn


class CrossAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=4096, seq_len=1024, bottleneck_dims=[64, 128, 256, 512, 768], num_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.query_tokens = nn.Parameter(torch.randn(seq_len, hidden_dim))

        self.token_decoder_heads = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(d, 1024),
                nn.GELU(),
                nn.Linear(1024, hidden_dim)
            )
            for d in bottleneck_dims
        })

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z, topk_idx, bottleneck_dim):
        B, k, _ = z.shape
        decoder = self.token_decoder_heads[str(bottleneck_dim)]
        v = decoder(z)  # [B, k, hidden_dim]

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, hidden_dim]
        recon, _ = self.attn(query=q, key=v, value=v)  # [B, seq_len, hidden_dim]
        recon = self.norm(recon)
        return recon