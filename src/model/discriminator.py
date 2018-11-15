# pylint: disable=E0401, E0302
import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_labels=1):
        super(Discriminator, self).__init__()

        self.num_labels = num_labels
        
        self.discriminator1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.discriminator2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.discriminator3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.discriminator4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.discriminator5 = nn.Sequential(
            nn.Linear(256*4*4, num_labels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.discriminator1(x)
        x = self.discriminator2(x)
        x = self.discriminator3(x)
        x = self.discriminator4(x)
        x = x.view(x.size(0), -1)
        x = self.discriminator5(x)

        return x