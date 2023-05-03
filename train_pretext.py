import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import torchvision.transforms as transforms


from .model.classification.global_module import GlobalModule
from .model.masked_autoencoders.mae import MAE
from .model.masked_autoencoders.masked_patch_prediction import MPP
from .data.dataset import ClassificationDataset


def main(config):
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

    # masked autoencoders: masked patch prediction
    mpp_trainer = MPP(
        transformer=global_module,
        patch_size=32,
        dim=1024,
        mask_prob=0.15,  # probability of using token in masked prediction task
        random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
        replace_prob=0.50,  # probability of replacing a token being used for mpp with the mask token
    )

    # dataloader
    dataset = ClassificationDataset(
        csv_file="power.csv", root_dir="test123", transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset, batch_size=32)

    # optimizer
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

    for epoch in range(100):
        for images, _ in train_loader:
            loss = mpp_trainer(images)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch {epoch}, loss {loss.item()}")

    # save your improved network
    torch.save(global_module.state_dict(), f"./pretrained-net-{epoch}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for image classification."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration file to use."
    )
    args = parser.parse_args()

    main(args.config)
