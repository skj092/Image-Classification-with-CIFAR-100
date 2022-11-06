from torchvision.datasets import MNIST
import torch, torchvision
import config
from torch.utils.data import DataLoader
from config import transform
from model import CNN
from config import load_checkpoint
from torchvision.datasets import CIFAR10


# evaluate
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(config.device)
            # images = images.view(images.size(0), 1, 28, 28)
            # images = images.reshape(images.shape[0], -1)
            labels = labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__=="__main__":
    # loading test MNIST dataset
    image_path = './data'
    test_ds = CIFAR10(root=image_path, train=False, transform=config.transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    # loading model
    model = CNN() 
    load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)
    print('model loaded successfully')
    model.to(config.device)

    evaluate(model, test_loader)