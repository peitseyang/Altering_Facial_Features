# pylint: disable=E0401, E0302
import torch
from torch import nn

class Discriminator(nn.Module):
	def __init__(self, numLabels=1):
		super(Discriminator, self).__init__()

		self.numLabels = numLabels
		self.discriminate = nn.Sequential(
			nn.Conv2d(3, 32, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, stride=2, padding=2),
			nn.ReLU()
		)
		self.flatten = nn.Sequential(
			nn.Linear(256 * 4 * 4, numLabels),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.discriminate(x)
		x = x.view(x.size(0), -1)
		x = self.flatten(x)
		
		return x


	# def save_params(self, exDir):
	# 	print('saving params...')
	# 	torch.save(self.state_dict(), join(exDir,'class_dis_params'))


	# def load_params(self, exDir):
	# 	print('loading params...')
	# 	self.load_state_dict(torch.load(join(exDir,'class_dis_params')))
