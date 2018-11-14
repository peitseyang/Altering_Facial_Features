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
from torch import nn
from torchvision.utils import save_image

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
parser.add_argument('--p2', default=1, type=float) # class to vae
parser.add_argument('--p3', default=1, type=float) # rec_class to vae
parser.add_argument('--p4', default=1, type=float) # aux_en to vae
parser.add_argument('--p5', default=1, type=float) # gen to vae

opts = parser.parse_args()


train_dataset = CelebA(label=opts.label, path=opts.path, transform=transforms.ToTensor())
test_dataset = CelebA(label=opts.label, path=opts.path, train=False, transform=transforms.ToTensor())
dataloader = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
}


cvae = CVAE(opts.latent_size, device).to(device)
dis = Discriminator().to(device)
aux = Aux(latent_size=opts.latent_size).to(device)
# print(cvae)
# print(dis)
# print(aux)


optimizer_cvae = torch.optim.RMSprop(cvae.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer_dis = torch.optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.momentum, weight_decay=opts.weight_decay)
optimizer_aux = torch.optim.RMSprop(aux.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

# losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[], 'aux':[], 'auxEnc':[]}
# Ns = len(trainLoader)*opts.batchSize  #no samples
# Nb = len(trainLoader)  #no batches

s_full_time = time()

for epoch in range(opts.epochs):
    cvae.train()
    dis.train()

    e_loss = 0
    e_rec_loss = 0
    e_kl_loss = 0
    e_class_loss = 0
    e_dis_loss = 0
    e_gen_loss = 0
    e_aux_loss = 0
    e_aux_en_loss = 0

    s_epoch_time = time()

    for i, data in enumerate(dataloader['train'], 0):
        x, y = data
        x = Variable(x).to(device)
        y = Variable(y).view(y.size(0),1).to(device)

        rec, mean, log_var, predict = cvae(x)
        # print('rec', rec)
        # print('mean', mean)
        # print('log_var', log_var)
        # print('predict', predict)
        z = cvae.reparameterization(mean, log_var)
        # print('z', z)
        rec_loss, kl_loss = cvae.loss(rec, x, mean, log_var)
        # print('rec_loss', rec_loss)
        # print('kl_loss', kl_loss)
        en_de_coder_loss = rec_loss + opts.p1 * kl_loss
        # print('en_de_coder_loss', en_de_coder_loss)

        ###
        loss = nn.BCELoss()
        class_loss = loss(predict.type_as(x), y.type_as(x))
        en_de_coder_loss += opts.p2 * class_loss

        rec_mean, rec_log_var, rec_predict = cvae.encoder(rec)
        # loss = nn.BCELoss()
        rec_class_loss = loss(rec_predict.type_as(x), y.type_as(x))
        en_de_coder_loss += opts.p3 * rec_class_loss

        aux_output = aux(z)
        # loss = nn.BCELoss()
        aux_en_loss = loss(aux_output.type_as(x), y.type_as(x))
        en_de_coder_loss -= opts.p4 * aux_en_loss

        aux_output = aux(z.detach())
        # loss = nn.BCELoss()
        aux_loss = loss(aux_output.type_as(x), y.type_as(x))
        ###

        ####
        dis_real = dis(x)
        dis_fake_rec = dis(rec.detach())
        randn_z = Variable(torch.randn(y.size(0), opts.latent_size)).to(device)
        randn_y = y.type_as(x)
        dis_fake_randn = dis(cvae.decoder(randn_y, randn_z).detach())
        label_fake = Variable(torch.Tensor(dis_real.size()).zero_()).type_as(dis_real)
        label_real = Variable(torch.Tensor(dis_real.size()).fill_(1)).type_as(dis_real)
        loss = nn.BCELoss(size_average=False)
        dis_loss = 0.3 * (loss(dis_real, label_real) + loss(dis_fake_randn, label_fake) + loss(dis_fake_randn, label_fake)) / dis_real.size(1)
        ####

        #####
        dis_fake_rec = dis(rec)
        dis_fake_randn = dis(cvae.decoder(randn_y, randn_z))
        gen_loss = 0.5 * (loss(dis_fake_rec, label_real) + loss(dis_fake_randn, label_real)) / dis_fake_rec.size(1)
        en_de_coder_loss += opts.p5 * gen_loss
        #####
        
        optimizer_cvae.zero_grad()
        en_de_coder_loss.backward()
        optimizer_cvae.step()
        optimizer_aux.zero_grad()
        aux_loss.backward()
        optimizer_aux.step()
        optimizer_dis.zero_grad()
        dis_loss.backward()
        optimizer_dis.step()


        e_loss += en_de_coder_loss.item()
        e_rec_loss += rec_loss.item()
        e_kl_loss += kl_loss.item()
        e_class_loss += class_loss.item()
        e_dis_loss += dis_loss.item()
        e_gen_loss += gen_loss.item()
        e_aux_loss += aux_loss.item()
        e_aux_en_loss += aux_en_loss.item()

        if i%100==1:
            i+=1
            print('[%d, %d] loss: %0.5f, gen: %0.5f, dis: %0.5f, bce: %0.5f, kl: %0.5f, aux: %0.5f, time: %0.3f' % \
               (epoch, i, e_loss/i, e_dis_loss/i, e_gen_loss/i, e_rec_loss/i, e_kl_loss/i, e_aux_loss/i, time() - s_epoch_time))
    if epoch % 5 == 4:
        cvae.eval()
        x, y = iter(dataloader['test']).next()
        x = Variable(x).to(device)
        y = Variable(y).view(y.size(0),1).to(device)

        if (y.data == 0).all():
            x0 = x
        else:
            zeroIdx = torch.nonzero(y.data)
            x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

        mean, log_var, y = cvae.encoder(x0)
        z = cvae.reparameterization(mean, log_var)

        y_smile = Variable(torch.LongTensor(np.ones(y.size(), dtype=int))).type_as(z)
        smile = cvae.decoder(y_smile, z).cpu()

        y_no_smile = Variable(torch.LongTensor(np.zeros(y.size(), dtype=int))).type_as(z)
        no_smile = cvae.decoder(y_no_smile, z).cpu()

        print('saving')
        save_image(x0.data, './ex/original.png')
        save_image(smile, './ex/smile_' + str(epoch) + '.png')
        save_image(no_smile, './ex/no_smile_' + str(epoch) + '.png')
