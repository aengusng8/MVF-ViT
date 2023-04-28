import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalModule:
    """
    Input:
        1. Image: original image, shape: (batch_size, 3, height, width)

    Output:
        1. global_K: shape: (batch_size, num_tokens, token_dim)
        2. global_V: shape: (batch_size, num_tokens, token_dim)
        3. global prediction: shape: (batch_size, 1)
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

        raise NotImplementedError
