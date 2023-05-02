import torch
import torch.nn as nn
import torch.nn.functional as F
from Local_Block2_5 import CNN_to_Tensors, Qlocal, LocalClassifier

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
        self.Local_CNN_1 = CNN_to_Tensors()
        self.Local_CNN_2 = CNN_to_Tensors()
        self.Local_CNN_3 = CNN_to_Tensors()
        self.local_cls = LocalClassifier(token_dim)
        self.Q_Local = Qlocal(token_dim)

    def forward(self, local_ROIs):
        eye_1, eye_2, nose, mouth = local_ROIs
        t1 = self.Local_CNN_1(eye_1)
        t2 = self.Local_CNN_1(eye_2)
        t3 = self.Local_CNN_2(nose)
        t4 = self.Local_CNN_3(mouth)

        local_Q = self.Q_Local([t1, t2, t3, t4])
        local_pred = self.local_cls(t1, t2, t3, t4)


        return local_Q, local_pred
