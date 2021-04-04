import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

class ASLNet(nn.Module):
    
    def __init__(self):
        super(ASLNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, 5)
        self.conv_2 = nn.Conv2d(16, 32, 5)
        self.conv_3 = nn.Conv2d(32, 64, 5)
        self.conv_4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 29)
        self.maxpool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        x = self.maxpool(F.relu(self.conv_1(x)))
        x = self.maxpool(F.relu(self.conv_2(x)))
        x = self.maxpool(F.relu(self.conv_3(x)))
        x = self.maxpool(F.relu(self.conv_4(x)))
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        
        return x