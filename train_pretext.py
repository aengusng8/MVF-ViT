import torch

from model.classification.global_module import GlobalModule
from model.classification.mae import MAE


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

    mae = MAE(
        encoder=global_module,
        masking_ratio=0.75,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=6,  # anywhere from 1 to 8
    )

    # TODO: add dataloader
    dataset = None
    dataloader = None

    for epoch in range(config["n_epochs"]):
        for batch_idx, (img, _) in enumerate(dataloader):
            loss = mae(img)
            loss.backward()
            print("self-supervised loss (reconstruction loss) ", loss.item())

    # save your improved vision transformer
    torch.save(global_module.state_dict(), "./mae-vit-epoch-{}.pt".format(epoch))


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
