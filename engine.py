import torch 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import config
from torch.optim.lr_scheduler import ExponentialLR



def train_model(model, train_dl, valid_dl, optimizer, loss_fn, scheduler=None):
    model.train()
    for epoch in range(config.EPOCHS):
        train_losses, train_acc, val_losses, val_accuracies = [], [], [], []
        loop = tqdm(train_dl)
        for xb, yb in loop:
            # move data to device
            xb, yb = xb.to(config.device), yb.to(config.device)
            # forward
            out = model(xb)
            # loss
            loss = loss_fn(out, yb)
            # backward
            loss.backward()
            optimizer.step()
            # zero grad
            optimizer.zero_grad()
            train_losses.append(loss.item()) 
            prediction = torch.argmax(out, dim=1)
            acc = accuracy_score(yb, prediction)
            train_acc.append(acc)
        if scheduler:
            scheduler.step()          
        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(config.device), yb.to(config.device)
                output = model(xb)
                loss = loss_fn(output, yb)
                prediction = torch.argmax(output, dim=1)
                accuracy = accuracy_score(yb, prediction)
                val_losses.append(loss)
                val_accuracies.append(accuracy)
        print(f'epoch={epoch}, train_loss = {sum(train_losses)/len(train_losses):.3f},\
            val_loss={sum(val_losses)/len(val_losses):.3f}, train_acc = {sum(train_acc)/len(train_acc):.3f}, \
                 val_acc={sum(val_accuracies)/len(val_accuracies):.3f}')