
import torch
import torch.nn as nn

class DeepMLPDecoder(nn.Module):
    def __init__(self, hidden_dim=4096, seq_len=1024, bottleneck_dims=[64, 128, 256, 512, 768]):
        super().__init__()
        self.seq_len = seq_len
        self.default_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        self.token_decoder_heads = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(d, 1024),
                nn.GELU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for d in bottleneck_dims
        })

    def forward(self, z, topk_idx, bottleneck_dim):
        B, k, _ = z.shape
        decoder = self.token_decoder_heads[str(bottleneck_dim)]
        decoded_topk = decoder(z)  # [B, k, hidden_dim]

        full_feat = self.default_token.expand(B, self.seq_len, -1).clone()
        for b in range(B):
            full_feat[b, topk_idx[b]] = decoded_topk[b]

        return full_feat

'''class FlexibleDecoder(nn.Module):
    def __init__(self, hidden_dim=4096, seq_len=1024, max_bottleneck_dim=512):
        super().__init__()
        self.seq_len = seq_len

        # Default token: small random init (helps avoid flat feature map)
        self.default_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        # Decoder heads for each bottleneck dimension
        self.token_decoder_heads = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(d, 1024),
                nn.ReLU(),
                nn.Linear(1024, hidden_dim)
            ) for d in [64, 128, 256, 512, 768]
        })

        # LayerNorm to stabilize final output
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z, topk_idx, bottleneck_dim):
        """
        z: [B, k, bottleneck_dim]
        topk_idx: [B, k]
        """
        B, k, _ = z.shape
        device = z.device

        decoder = self.token_decoder_heads[str(bottleneck_dim)]
        decoded_topk = decoder(z)  # [B, k, hidden_dim]

        # Initialize full feature map with default token
        full_feat = self.default_token.expand(B, self.seq_len, -1).clone()

        # Replace top-k locations
        for b in range(B):
            full_feat[b, topk_idx[b]] = decoded_topk[b]

        # Normalize to stabilize for downstream layers
        return self.norm(full_feat)'''