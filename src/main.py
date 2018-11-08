# pylint: disable=E0401
import argparse
from time import time
import numpy as np

from data.dataloader import CelebA
from model.cvae import CVAE
from model.discriminator import Discriminator
from model.aux import Aux

from torchvision import transforms

import torch
from torch.autograd import Variable

print('pytorch version : ' + str(torch.__version__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--label', default='Smiling', type=str)
parser.add_argument('--path', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/celebA/', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--latent_size', default=200, type=int)

parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--weight_decay', default=0.01, type=int)

parser.add_argument('--epochs', default=40, type=int)

parser.add_argument('--p1', default=1, type=float) # kl to vae

opts = parser.parse_args()


train_dataset = CelebA(label=opts.label, path=opts.path, transform=transforms.ToTensor())
test_dataset = CelebA(label=opts.label, path=opts.path, train=False, transform=transforms.ToTensor())
dataloader = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
}


cvae = CVAE(opts.latent_size, device).to(device)
dis = Discriminator().to(device)
# aux = Aux(latent_size=opts.latent_size)
# print(cvae)
# print(dis)
# print(aux)


optimizer_cvae = torch.optim.RMSprop(cvae.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer_dis = torch.optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.momentum, weight_decay=opts.weight_decay)
# optimizer_aux = torch.optim.RMSprop(aux.parameters, lr=opts.lr, weight_decay=opts.weight_decay)

# losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[], 'aux':[], 'auxEnc':[]}
# Ns = len(trainLoader)*opts.batchSize  #no samples
# Nb = len(trainLoader)  #no batches

s_full_time = time()

for epoch in range(opts.epochs):
    cvae.train()
    dis.train()

    s_epoch_time = time()

    for i, data in enumerate(dataloader['train'], 0):
        x, y = data
        x = Variable(x).to(device)
        # Variable(y).view(y.size(0),1).type_as(x)
        y = Variable(y).view(y.size(0),1).to(device)

        rec, mean, log_var, predict = cvae(x)
        z = cvae.reparameterization(mean, log_var)
        rec_loss, kl_loss = cvae.loss(rec, x, mean, log_var)
        en_de_coder_loss = rec_loss + opts.p1 * kl_loss
        


        break
