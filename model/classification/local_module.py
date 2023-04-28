import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalModule:
    """
    Input:
        1. local_ROIs: eyes x2, nose x1, mouth x1, shape

    Output:
        1. local_Q: shape: (batch_size, num_tokens, token_dim)
        2. local prediction: shape: (batch_size, 1)
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

    def forward(self, local_ROIs):
        eyes, nose, mouth = local_ROIs

        raise NotImplementedError
        return local_Q, local_prediction
