# pylint: disable=E0401, E0302
import torch
from torch import nn

class Aux(nn.Module):
	def __init__(self, latent_size, num_labels=1):
		super(Aux, self).__init__()

		self.latent_size = latent_size
		self.num_labels = num_labels

		self.aux = nn.Sequential(
			nn.Linear(latent_size, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, num_labels),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.aux(x)

		return x
