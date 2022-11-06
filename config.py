from torchvision import transforms
import torch 

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "my_checkpoint.pth.tar"

transform = transforms.Compose([
    transforms.ToTensor()
])

# save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# load checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

# Learning rate scheduler
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer  