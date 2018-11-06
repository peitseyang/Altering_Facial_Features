# pylint: disable=E0401, E0302
import torch
from torch import nn

class Aux(nn.Module):
    def __init__(self, latent_size, num_labels=1):
        super(Aux, self).__init__()

        self.latent_size = latent_size
        self.num_labels = num_labels

        self.aux1 = nn.Sequential(
            nn.Linear(latent_size, 1000),
            nn.ReLu()
        )
        self.aux2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLu()
        )
        self.aux3 = nn.Sequential(
            nn.Linear(1000, num_labels),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.aux1(z)
        z = self.aux2(z)
        z = self.aux3(z)

        return z

    # def loss(self, pred, target):
	# 	return F.nll_loss(pred, target)