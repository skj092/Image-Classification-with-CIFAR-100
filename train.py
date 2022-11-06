from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torch
import numpy as np 
from model import CNN
import config
from config import load_checkpoint, save_checkpoint, lr_scheduler
import torch.nn as nn 
from engine import train_model
from  config import transform
import os 
from torch.optim import lr_scheduler

if __name__=="__main__":
    image_path = './data'
    train = CIFAR10(root=image_path, train=True, transform=config.transform, download=True)
    test = CIFAR10(root=image_path, train=False, transform=config.transform, download=True)

    # # # creating a validation set
    train_ds = Subset(train, np.arange(0, 45000))
    valid_ds = Subset(train, range(0, 5000))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE)
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE)
    

    model = CNN()
    model.to(config.device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # load checkpoint
    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)


    # train
    train_model(model, train_loader,valid_loader, optimizer=optimizer, loss_fn=loss_fn)
    
    # save checkpoint
    if config.SAVE_MODEL:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)