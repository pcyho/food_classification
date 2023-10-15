import numpy as np
import torch.optim
import torchvision.transforms as transform

from data import Data
from model import Module
from torch import mode, nn
from torch.utils.data import DataLoader

train_transform = transform.Compose([
    transform.ToPILImage(),
    transform.RandomHorizontalFlip(),
    transform.ToTensor()
])

test_transform = transform.Compose([
    transform.ToPILImage(),
    transform.ToTensor(),
])

model = Module()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

train_data = Data("./dataset/training", transform=train_transform)
validation_data = Data("./dataset/validation", train_transform)
train_loader = DataLoader(train_data, shuffle=True)
validation_loader = DataLoader(validation_data, shuffle=True)

for epoch in range(10):
    print(f"-------{epoch}-------")
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for batch, (x, y) in enumerate(train_loader):
        train_pred = model(x)
        batch_loss = loss_fn(train_pred, y)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_acc += np.sum(
            np.argmax(train_pred.data.numpy(), axis=1) == y.numpy())
        train_loss += batch_loss.item()

        if batch % 100 == 0:
            print(
                f"loss: {train_loss / train_data.__len__():3.6f}  acc: {train_acc / train_data.__len__():3.6f}"
            )
    model.eval()
