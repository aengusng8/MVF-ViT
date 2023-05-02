import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import pair, Transformer


class FusionModule(nn.Module):
    """
    Get input from local module and global module, then fuse them together
    Input:
        1. local_Q: from local module, shape: (batch_size, num_tokens, token_dim)
        2. global_tokens (= K = V): from global module, shape: (batch_size, num_tokens, token_dim)

    Output:
        1. fusion_prediction: shape: (batch_size, 1)

    """

    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout
        )

    def forward(self, local_Q, global_tokens, cls_tokens):
        x, _ = self.cross_attention(
            query=local_Q, key=global_tokens, value=global_tokens
        )

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":

    def test_fusion_module():
        batch_size = 8
        num_tokens = 16
        token_dim = 1024

        # Define input tensor
        local_Q = torch.randn(batch_size, num_tokens, token_dim)
        global_tokens = torch.randn(batch_size, num_tokens, token_dim)
        cls_tokens = torch.randn(batch_size, 1, token_dim)

        # Create fusion module
        fusion_module = FusionModule(
            num_classes=1,
            dim=1024,
            depth=3,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
        )

        # Apply fusion module to input tensor
        output_tensor = fusion_module(local_Q, global_tokens, cls_tokens)

        # Check output tensor shape
        print(output_tensor.shape)

    # test_fusion_module()
