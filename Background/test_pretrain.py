'''



'''
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import lightning as pl
import torch
import torch.nn as nn
import torchvision

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms


def main():
    simclr_transform = transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToImage(**spt.data.static.CIFAR10),
            ),
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),  # 最小能裁到原图面积的 8%
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomSolarize(threshold=0.5, p=0.2),  #颜色反转风格
                transforms.ToImage(**spt.data.static.CIFAR10),
            ),
        ]
    )

    cifar_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True
    )

    train_dataset = spt.data.FromTorchDataset(
        cifar_train,
        names=["image", "label"],
        transform=simclr_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    data = spt.data.DataModule(train=train_loader)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True) #从 torchvision 加载一个适合低分辨率图像的 ResNet-18，作为 SimCLR 的特征提取主干网络。


    backbone.fc = nn.Identity()


    # 把 backbone 提取出来的特征，再映射到一个更适合做对比学习的空间里。
    projector = nn.Sequential(
        nn.Linear(512, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 256),
    )

    module = spt.Module(
        backbone=backbone, #这个就是把你前面定义好的主干网络传进去。
        projector=projector, #这个就是把你刚才定义的 MLP 投影头传进去。
        forward=forward.simclr_forward, #这个模块训练时，前向传播逻辑用 stable_pretraining 里现成的 simclr_forward
        simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
        optim={
            "optimizer": {"type": "Adam", "lr": 1e-3, "weight_decay": 1e-6},
            "scheduler": {"type": "CosineAnnealingLR"},
            "interval": "epoch",  #每个 epoch 更新一次 scheduler
        },
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="mps",
        devices=1,
        precision=32,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        val_check_interval=0,
        limit_val_batches=0,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()