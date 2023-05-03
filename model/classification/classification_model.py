import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_module import FusionModule
from global_module import GlobalModule
from local_module import LocalModule
from roi_cropping.crop_roi import roi_crop


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

    def __init__(
        self,
        num_classes,
        fs_dim,
        fs_depth,
        fs_heads,
        fs_mlp_dim,
        fs_pool="cls",
        fs_channels=3,
        fs_dim_head=64,
        fs_dropout=0.0,
        fs_emb_dropout=0.0,
        lc_token_dim=1024,
        gb_image_size=112,
        gb_patch_size=28,
        gb_num_classes=1,
        gb_dim=1024,
        gb_depth=6,
        gb_heads=4,
        gb_mlp_dim=1024,
        gb_dropout=0.1,
        gb_emb_dropout=0.1,
    ):
        super().__init__()
        self.local_module = LocalModule(lc_token_dim)
        self.global_module = GlobalModule(
            gb_image_size,
            gb_patch_size,
            gb_num_classes,
            gb_dim,
            gb_depth,
            gb_heads,
            gb_mlp_dim,
            gb_dropout,
            gb_emb_dropout,
        )
        self.fusion_module = FusionModule(
            num_classes,
            fs_dim,
            fs_depth,
            fs_heads,
            fs_mlp_dim,
            fs_pool,
            fs_channels,
            fs_dim_head,
            fs_dropout,
            fs_emb_dropout,
        )

    def forward(self, local_ROIs, image):
        # get rois
        local_ROIs = roi_crop(image_file=image)

        # 1. Local module
        local_Q, local_pred = self.local_module(local_ROIs)

        # 2. Global module
        global_tokens, cls_tokens, global_pred = self.global_module(image)

        # 3. Fusion module
        fusion_pred = self.fusion_module(local_Q, global_tokens, cls_tokens)

        # raise NotImplementedError
        return local_pred, global_pred, fusion_pred
