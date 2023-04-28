import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationModel(nn.Module):
    """
    This is the classification model for our task consisting of 3 modules:
        1. Local module
        2. Global module
        3. Fusion module

    Input:
        1. local_ROIs: eyes x2, nose x1, mouth x1, shape
        2. Image: original image, shape: (batch_size, 3, height, width)

    Output:
        1. Local prediction: shape: (batch_size, 1)
        2. Global prediction: shape: (batch_size, 1)
        3. Fusion prediction: shape: (batch_size, 1)
    """

    def __init__(self,):
        super().__init__()

        raise NotImplementedError

    def forward(self, local_ROIs, image):
        # 1. Local module

        # 2. Global module

        # 3. Fusion module

        raise NotImplementedError
        return local_prediction, global_prediction, fusion_prediction
