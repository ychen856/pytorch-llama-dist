import torch
import torch.nn as nn

class FeatureDecoder(nn.Module):
    def __init__(self, vocab_size=32000, embedding_dim=4096, output_dim=4096, seq_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.seq_len = seq_len
        self.output_dim = output_dim

        # 將 k 個 token 聚合回 seq_len × output_dim 的 feature vector
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, seq_len * output_dim)  # flatten target
        )

    def forward(self, token_ids):  # token_ids: [batch_size, k]
        embedded = self.embedding(token_ids)  # [batch, k, embed_dim]
        pooled = embedded.mean(dim=1)  # [batch, embed_dim]
        recon_flat = self.decoder(pooled)  # [batch, seq_len * output_dim]
        recon_feature = recon_flat.view(-1, self.seq_len, self.output_dim)  # [batch, seq_len, 4096]
        return recon_feature
