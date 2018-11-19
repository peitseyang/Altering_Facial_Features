# pylint: disable=E0401, E0302
import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

import os
from os.path import join

class CVAE(nn.Module):

	def __init__(self, latent_size, device, num_labels=1):
		super(CVAE, self).__init__()
		#define layers here

		self.latent_size = latent_size
		self.device = device
		self.num_labels = num_labels
		
		self.encode = Encoder(latent_size, num_labels)
		self.decode = Decoder(latent_size, num_labels)

	def forward(self, x):
		mean, log_var, y = self.encode(x)
		z = self.reparameterization(mean, log_var)
		rec = self.decode(y, z)

		return rec, mean, log_var ,y

	def reparameterization(self, mean, log_var):
		e = Variable(torch.randn(log_var.size(0), self.latent_size)).to(self.device)
		return mean + torch.exp(log_var/2) * e

	def loss(self, rec, x, mean, log_var):
		loss = nn.BCELoss(size_average=False)
		bce = loss(rec, x)
		# KL divergence is a useful distance measure for continuous distributions and is often useful when performing direct regression over the space of (discretely sampled) continuous output distributions.
		# As with NLLLoss, the input given is expected to contain log-probabilities, however unlike ClassNLLLoss, input is not restricted to a 2D Tensor, because the criterion is applied element-wise.
		# This criterion expects a target Tensor of the same size as the input Tensor.Ｆ
		kl = 0.5 * torch.sum(mean ** 2 + torch.exp(log_var) - 1. - log_var)

		return bce / (64 * 64), kl / self.latent_size

	def save_params(self, path):
		print('saving params...')
		torch.save(self.state_dict(), join(path, 'cvae1_params'))

	def load_params(self, path):
		print('loading params...')
		self.load_state_dict(torch.load(join(path, 'cvae1_params'), map_location='cpu'))

class Encoder(nn.Module):
	def __init__(self, latent_size, num_labels):
		super(Encoder, self).__init__()

		self.encode = nn.Sequential(
			nn.Conv2d(3, 32, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, stride=2, padding=2),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, stride=2, padding=2),
			nn.ReLU()
		)
		self.mean = nn.Linear(256*4*4, latent_size)
		self.log_var = nn.Linear(256*4*4, latent_size)
		self.y = nn.Sequential(
			nn.Linear(256*4*4, num_labels),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.encode(x)
		x = x.view(x.size(0), -1)
		mean = self.mean(x)
		log_var = self.log_var(x)
		y = self.y(x.detach())

		return mean, log_var, y

class Decoder(nn.Module):
	def __init__(self, latent_size, num_labels):
		super(Decoder, self).__init__()

		self.decode1 = nn.Sequential(
			nn.Linear(latent_size+1, 256*4*4),
			nn.ReLU(),
		)
		self.decode2 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1 ,output_padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1 ,output_padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1 ,output_padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1 ,output_padding=1),
			nn.Sigmoid()
		)

	def forward(self, y, z):
		x = torch.cat([y, z], dim=1)
		x = self.decode1(x)
		x = x.view(z.size(0), -1, 4, 4)
		x = self.decode2(x)

		return x
