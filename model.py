import torch
from torch import nn as nn
from torch.nn.modules.flatten import Flatten


class Module(nn.Module):
    def __init__(self):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input [3, 128, 128]
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64,64,64]
            
            nn.Conv2d(64, 128, 3, 1, 1),  # [128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [128,32,32]
            
            nn.Conv2d(128, 256, 3, 1, 1),  # [256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [256,16,16]
            
            nn.Conv2d(256, 512, 3, 1, 1),  # [512,16,16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [512,8,8]
            
            nn.Conv2d(512, 512, 3, 1, 1),  # [512,8,8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [512,4,4]
            Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11))

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = Module()
    input = torch.ones((64, 3, 128, 128))
    output = model(input)
    print(output.shape)
