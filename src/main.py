# pylint: disable=E0401
import argparse

from data.dataloader import CelebA
from model.cvae import CVAE
from model.discriminator import Discriminator
from model.aux import Aux

from torchvision import transforms
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--label', default='Smiling', type=str)
parser.add_argument('--path', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/celebA/', type=str)
parser.add_argument('--latent_size', default=200, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--weight_decay', default=0.01, type=int)

opts = parser.parse_args()


train_dataset = CelebA(label=opts.label, path=opts.path, transform=transforms.ToTensor())
test_dataset = CelebA(label=opts.label, path=opts.path, train=False, transform=transforms.ToTensor())
dataloader = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
}


cvae = CVAE(latent_size=opts.latent_size).to(device)
dis = Discriminator().to(device)
# aux = Aux(latent_size=opts.latent_size)
print(cvae)
print(dis)
# print(aux)


optimizer_cvae = torch.optim.RMSprop(cvae.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer_dis = torch.optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.momentum, weight_decay=opts.weight_decay)
# optimizer_aux = torch.optim.RMSprop(aux.parameters, lr=opts.lr, weight_decay=opts.weight_decay)

# losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[], 'aux':[], 'auxEnc':[]}
# Ns = len(trainLoader)*opts.batchSize  #no samples
# Nb = len(trainLoader)  #no batches