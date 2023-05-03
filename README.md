# MVF-ViT (Multi-View Fusion Vision Transformer)

## Method
MVF-ViT is a deep learning model for visual recognition tasks that combines global and local information using cross-attention. This cross-attention mechanism allows the model to attend to relevant global features based on the local features and vice versa, enabling effective information fusion. 

The model comprises three modules: a global module, a local module, and a fusion module.

![model overview drawio (1)](https://user-images.githubusercontent.com/67547213/235863449-3b1b4ce3-33f2-4ac8-83a4-3de67aa57bf6.png)

### Global module
The global module predicts using global information only and outputs both the global prediction and the global features. This module is pre-trained using the self-supervision training method called  [Masked Image Modeling (MIM)](https://arxiv.org/abs/2111.06377).
1. MIM self-supervision training
- Input: the masked image 
- Ouput: the reconstructed image
2. The pre-trained global module is then used as part of the MVF-ViT model for downstream tasks.
- Input: the original image
- Ouput: the global prediction, the global features, and the class token (in tokens matrix)

### Local module
The local module uses a low-capacity, memory-efficient network to identify the most informative regions of the image. These regions are then processed to produce local features, which are embeddings of regions of interest (ROIs).
- Input: the original image
- Output: the local feature which are the embedings of region of interest (often abbreviated ROI)

### Fusion module
The fusion module combines the local and global features along with the class token from the global module to produce the final prediction. This module uses cross-attention, where the query is the local feature, and the key and the value are the global feature.
- Input: the local features (from local module), the global features and the class token (from the global module)
- Output: the fusion prediction

## Acknowledgments
- We thank He et al. for introducing the Masked Image Modeling (MIM) self-supervision training method in their paper "Masked Autoencoders Are Scalable Vision Learners", which was used to pre-train the global module in our MVF-ViT model. (https://arxiv.org/abs/2012.12877) 
- We also acknowledge Shen et al. for their paper "An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization", which inspired this work. (https://arxiv.org/abs/2002.07613)
