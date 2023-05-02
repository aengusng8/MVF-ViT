import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit import pair, Transformer
from mae import MAE



class GlobalModule(nn.Module):
    """
    Input:
        1. Image: original image, shape: (batch_size, 3, height, width)

    Output:
        1. global_tokens: shape: (batch_size, num_tokens, token_dim)
        2. global_prediction: shape: (batch_size, 1)
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
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
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img, get_global_tokens=True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        global_tokens = x[:, 1:, :]

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        global_prediction = self.mlp_head(x)

        if get_global_tokens:
            return global_tokens, global_prediction
        else:
            return global_prediction


if __name__ == "__main__":

    def test_global_module():
        global_module = GlobalModule(
            image_size=112,
            patch_size=28,
            num_classes=1,
            dim=1024,
            depth=6,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
        )

        
        images = torch.randn(8, 3, 112, 112)

        preds = global_module(images)
        print("debug: preds.shape ", preds.shape)

        # calculate number of parameters
        print(sum(p.numel() for p in global_module.parameters() if p.requires_grad))

        mae = MAE(
            encoder = global_module,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
        )

        loss = mae(images)
        loss.backward()
        print("debug: self-supervised loss ", loss.item())

    test_global_module()
