import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule:
    """
    Get input from local module and global module, then fuse them together
    Input:
        1. local_Q: from local module, shape: (batch_size, num_tokens, token_dim)
        2. global_K: from global module, shape: (batch_size, num_tokens, token_dim)
        3. global_V: from global module, shape: (batch_size, num_tokens, token_dim)

    Output:

    """

    def __init__(self, dropout, token_dim, num_classes=1):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_dim // 2, token_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_dim // 4, num_classes),
        )

    def forward(self, local_Q, global_K, global_V):
        assert local_Q.shape == global_K.shape == global_V.shape


if __name__ == "__main__":
    assert (1, 2, 3) == (1, 2, 3)
    print("ok")
