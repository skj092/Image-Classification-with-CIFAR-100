from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch
import numpy as np
from model import CNN
import config
from config import lr_scheduler, load_last_checkpoint
import torch.nn as nn
from engine import train_model
import logging
import wandb
import os

if __name__ == "__main__":
    image_path = './data'
    train = CIFAR10(root=image_path, train=True,
                    transform=config.transform, download=True)
    test = CIFAR10(root=image_path, train=False,
                   transform=config.transform, download=True)

    # # # creating a validation set
    train_ds = Subset(train, np.arange(0, 45000))
    valid_ds = Subset(train, range(0, 5000))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE)
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE)

    model = CNN()
    model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if config.LOAD_MODEL:
        model, epoch = load_last_checkpoint(
            config.CHECKPOINT_DIR, model, optimizer)
    if config.WANDB:
        print("Logging in wandb")
        os.environ["WANDB_API_KEY"] = "97b5307e24cc3a77259ade3057e4eea6fd2addb0"
        wandb.init(project="cifar10", name="cifar10 pipeline test")
        wandb.watch(model, log="all")
        logger = logging.getLogger("wandb")
    else:
        logger = None

    # train
    train_model(model, train_loader, valid_loader,
                optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
