import glob
from torchvision import transforms
import torch
import os

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DIR = 'checkpoints'
WANDB = False

transform = transforms.Compose([
    transforms.ToTensor()
])

# CIFAR-10 classes
CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# save checkpoint


def save_checkpoint(checkpoint: dict, dir: str):
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    os.makedirs(dir, exist_ok=True)
    filename = f'{dir}/epoch={epoch}-val_loss={val_loss}.pth'
    state = {
        'state_dict': checkpoint['state_dict'],
        'optimizer': checkpoint['optimizer'],
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(state, filename)


# load checkpoint


def load_last_checkpoint(dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    checkpoint_files = glob.glob(f'{dir}/*.pth')

    if len(checkpoint_files) == 0:
        print("No checkpoints found loading model from scratch")
        return model, 0

    # Sort files by modification time (most recent first)
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    print(f"=> Loading the most recent checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(
        latest_checkpoint, map_location=torch.device('cpu'), weights_only=True)

    model.load_state_dict(checkpoint['state_dict'])
    print(
        f"Loaded model state from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']}")

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded optimizer state")

    return model, checkpoint['epoch']

# Learning rate scheduler


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
