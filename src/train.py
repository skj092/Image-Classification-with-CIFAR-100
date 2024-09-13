from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch
import numpy as np
from model import CNN, EfficientNet, ResNet, DenseNet
import config
from config import load_last_checkpoint
import torch.nn as nn
from engine import train_model
import logging
import wandb
from dotenv import load_dotenv
from torch.optim import lr_scheduler


load_dotenv()

def main():
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

    if config.model_name == "cnn":
        model = CNN()
    elif config.model_name == "efficientnet":
        model = EfficientNet()
    elif config.model_name == "resnet":
        model = ResNet()
    elif config.model_name == "densenet":
        model = DenseNet()
    else:
        raise ValueError("Model not found")
    model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    resume = False

    if config.LOAD_MODEL:
        model, epoch = load_last_checkpoint(
            config.CHECKPOINT_DIR, model, optimizer)
        resume = True
    if config.WANDB:
        print("Logging in wandb")
        conf = {
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
        }
        wandb.init(project="cifar10",
                   name="cifar10 pipeline test", config=conf, resume=resume)
        wandb.watch(model, log="all")
        logger = logging.getLogger("wandb")
    else:
        logger = None

    # train
    train_model(model, train_loader, valid_loader,
                optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    config.WANDB = args.wandb
    config.BATCH_SIZE = args.batch_size
    config.model_name = args.model_name
    main()
