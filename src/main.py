# pylint: disable=E0401, E0302

import argparse
from time import time
import numpy as np

from data.dataloader import CelebA
from model.cvae import CVAE
from model.discriminator import Discriminator
from model.classifier import Classifier
from model.discriminator import Discriminator as CLASSIFIERS

from torchvision import transforms
import torch.nn.functional as F

import torch
from torch.autograd import Variable
from torch import nn
from torchvision.utils import save_image

import os
from os.path import join
from PIL import Image

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

EPSILON = 1e-6

import itertools
import threading
import sys
from time import sleep

done = False

def animate():
    for c in itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠴', '⠦', '⠧', '⠇', '⠏']):
        if done:
            break
        sys.stdout.write('\r' + c + ' ')
        sys.stdout.flush()
        sleep(0.1)
    sys.stdout.write('\rDone!     ')


t = threading.Thread(target=animate)
t.start()

# python3 celeba_info_cVAEGAN.py --alpha 0.2 --batch_size 32 --beta 0 --delta 0.1 --fSize 32 --epochs 45 --rho 0.1
# python3 main.py --epochs 45 --alpha 0.2 --delta 0.1 --rho 0.1

def label_switch(x,y,cvae,exDir=None): #when y is a unit not a vector
    print('switching label')
    # #get x's that have smile
    # if (y.data == 0).all(): #if no samples with label 1 use all samples
    #     x0 = Variable(x)
    # else:
    #     zeroIdx = torch.nonzero(y.data)
    #     x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

    #get z
    mu, logVar, y = cvae.encode(x, y)
    z = cvae.reparameterization(mu, logVar)

    ySmile = Variable(torch.LongTensor(np.ones(y.size(), dtype=int))).type_as(z)
    smileSamples = cvae.decode(ySmile, z)    
    

    yNoSmile = Variable(torch.LongTensor(np.zeros(y.size(), dtype=int))).type_as(z)
    noSmileSamples = cvae.decode(yNoSmile, z)
    
    if exDir is not None:
        print('saving rec w/ and w/out label switch to', join(exDir,'rec.png'),'... ')
        save_image(x.data, join(exDir, 'original.png'))
        save_image(smileSamples.cpu().data, join(exDir,'rec_1.png'))
        save_image(noSmileSamples.cpu().data, join(exDir,'rec_0.png'))

    return smileSamples, noSmileSamples

# def binary_class_score(pred, target, thresh=0.5):
#     predLabel = torch.gt(pred, thresh)
#     classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
#     return  classScoreTest.float().sum()/target.size(0)

def evaluate(cvae, test_data, exDir, e=1):  #e is the epoch
    cvae.eval()

    test_x, test_y = iter(test_data).next()
    test_x = Variable(test_x).to(cvae.device)
    test_y = Variable(test_y).view(test_y.size(0),1).to(cvae.device)

    z = Variable(torch.randn(test_x.size(0), opts.latent_size)).to(cvae.device)

    y_0 = Variable(torch.ones(test_y.size())).type_as(test_x)
    samples = cvae.decode(y_0, z).cpu()
    save_image(samples.data, join(exDir,'zero_epoch'+str(e)+'.png'))

    y_1 = Variable(torch.zeros(test_y.size())).type_as(test_x)
    samples = cvae.decode(y_1, z).cpu()
    save_image(samples.data, join(exDir,'one_epoch'+str(e)+'.png'))

    test_rec, test_mean, test_log_var, test_predict = cvae(test_x, test_y)


    test_bce_loss, test_kl_loss = cvae.loss(test_rec, test_x, test_mean, test_log_var)
    predict_label = torch.floor(test_predict)

    # classScoreTest= binary_class_score(predict_label, test_y, thresh=0.5)
    # print('classification test:', classScoreTest.data[0])

    save_image(test_x.data, join(exDir,'input.png'))
    save_image(test_rec.data, join(exDir,'output_'+str(e)+'.png'))

    rec1, rec0 = label_switch(test_x.data, test_y, cvae, exDir=exDir)


    return (test_bce_loss).data[0]/test_x.size(0)
    # , classScoreTest.data[0]

if __name__=='__main__':

    print('pytorch version : ' + str(torch.__version__))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='Smiling', type=str)
    parser.add_argument('--path', default='/home/csie-owob/alexyang/yerin/Altering_Facial_Features/src/data/celebA/', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--latent_size', default=200, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--b1', default=0.5, type=float)
    parser.add_argument('--b2', default=0.999, type=float)

    parser.add_argument('--alpha', default=1, type=float, help='p1') #weight on the KL divergance
    parser.add_argument('--rho', default=1, type=float, help='p2') #weight on the class loss for the vae
    # parser.add_argument('--beta', default=1, type=float, help='p3')  #weight on the rec class loss to update VAE
    parser.add_argument('--gamma', default=1, type=float, help='p4') #weight on the classifier enc loss
    parser.add_argument('--delta', default=1, type=float, help='p5') #weight on the adversarial loss


    opts = parser.parse_args()


    train_dataset = CelebA(label=opts.label, path=opts.path, transform=transforms.ToTensor())
    test_dataset = CelebA(label=opts.label, path=opts.path, train=False, transform=transforms.ToTensor())
    dataloader = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
    }


    cvae = CVAE(opts.latent_size, device).to(device)
    dis = Discriminator().to(device)
    classifier = Classifier(opts.latent_size).to(device)
    classer = CLASSIFIERS().to(device)

    print(cvae)
    print(dis)
    print(classifier)


    optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=opts.lr,  betas=(opts.b1, opts.b2), weight_decay=opts.weight_decay)
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=opts.lr,  betas=(opts.b1, opts.b2), weight_decay=opts.weight_decay)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=opts.lr,  betas=(opts.b1, opts.b2), weight_decay=opts.weight_decay)

    i = 1
    while os.path.isdir('./ex/' + str(i)):
        i += 1
    os.mkdir('./ex/' + str(i))
    output_path = './ex/' + str(i)
    # print('Outputs will be saved to:', output_path)
    # f = open(join(output_path,'opts.txt'),'w')
    # saveOpts =''.join(''.join(str(opts).split('(')[1:]).split(')')[:-1]).replace(',','\n')
    # f.write(saveOpts)
    # f.close()

    losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'class':[], 'classifier':[]}
    Ns = len(dataloader['train'])*opts.batch_size
    Nb = len(dataloader['train'])


    full_time = time()

    for e in range(opts.epochs):
        cvae.train()
        dis.train()

        e_loss = 0
        e_rec_loss = 0
        e_kl_loss = 0
        e_class_loss = 0
        e_dis_loss = 0
        e_gen_loss = 0
        e_classifier_loss = 0
        e_classifier_en_loss = 0

        epoch_time = time()

        for i, data in enumerate(dataloader['train'], 0):
            x, y = data
            x = Variable(x).to(device)
            y = Variable(y).view(y.size(0),1).to(device)


            rec, mean, log_var, c = cvae(x, y)
            z = cvae.reparameterization(mean, log_var)
            rec_loss, kl_loss = cvae.loss(rec, x, mean, log_var)
            en_de_coder_loss = rec_loss + opts.alpha * kl_loss

            # loss = nn.BCELoss()
            # print(c)
            # class_loss = loss(c.type_as(x), y.type_as(x))
            # en_de_coder_loss += opts.rho * class_loss

            # rec_mean, rec_log_var, rec_predict = cvae.encode(rec)
            # rec_class_loss = loss(rec_predict.type_as(x), y.type_as(x))
            # en_de_coder_loss += opts.beta * rec_class_loss

            classifierY = classifier(z)
            classifier_en_loss = loss(classifierY.type_as(x), y.type_as(x))  
            en_de_coder_loss -= opts.gamma * classifier_en_loss
            classifierY = classifier(z.detach())
            classifier_loss = loss(classifierY.type_as(x), y.type_as(x))

            dis_real = dis(x)
            dis_fake_rec = dis(rec.detach())
            randn_z = Variable(torch.randn(y.size(0), opts.latent_size)).to(device)
            dis_fake_randn = dis(cvae.decode(y.type_as(x), randn_z).detach())
            label_fake = Variable(torch.Tensor(dis_real.size()).zero_()).type_as(dis_real)
            label_real = Variable(torch.Tensor(dis_real.size()).fill_(1)).type_as(dis_real)
            loss = nn.BCELoss(size_average=False)
            dis_loss = 0.3 * (loss(dis_real, label_real) + loss(dis_fake_rec, label_fake) + loss(dis_fake_randn, label_fake)) / dis_real.size(1)

            dis_fake_rec = dis(rec)
            dis_fake_randn = dis(cvae.decode(y.type_as(x), randn_z))
            gen_loss = 0.5 * (loss(dis_fake_rec, label_real) + loss(dis_fake_randn, label_real)) / dis_fake_rec.size(1)
            en_de_coder_loss += opts.delta * gen_loss


            optimizer_cvae.zero_grad()
            en_de_coder_loss.backward()
            optimizer_cvae.step()
            optimizer_classifier.zero_grad()
            classifier_loss.backward()
            optimizer_classifier.step()
            optimizer_dis.zero_grad()
            dis_loss.backward()
            optimizer_dis.step()


            e_loss += en_de_coder_loss.item()
            e_kl_loss += kl_loss.item()
            e_rec_loss += rec_loss.item()
            e_gen_loss += gen_loss.item()
            e_dis_loss += dis_loss.item()
            e_class_loss += class_loss.item()
            e_classifier_loss += classifier_loss.item()
            e_classifier_en_loss += classifier_en_loss.item()

            if i%100==1:
                i+=1
                print('[%d, %d] loss: %0.5f, bce: %0.5f, kl: %0.5f, gen: %0.5f, dis: %0.5f, classifier: %0.5f, time: %0.3f' % (e, i, e_loss/i, e_rec_loss/i, e_kl_loss/i, e_gen_loss/i, e_dis_loss/i,  e_classifier_loss/i, time() - epoch_time))


        normbceLossTest = evaluate(cvae, dataloader['test'], output_path, e=e)

        cvae.save_params(path=output_path)

        losses['total'].append(e_loss/Ns)
        losses['kl'].append(e_kl_loss/Ns)
        losses['bce'].append(e_rec_loss/Ns)
        losses['dis'].append(e_dis_loss/Ns)
        losses['gen'].append(e_gen_loss/Ns)
        losses['class'].append(e_class_loss/Ns)
        losses['classifier'].append(e_classifier_loss/Ns)


        fig1 = plt.figure()
        for key in losses:
            noPoints = len(losses[key])
            factor = float(noPoints)/(e+1)
            plt.plot(np.arange(len(losses[key]))/factor,losses[key], label=key)

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('loss')
        fig1.savefig(join(output_path, 'loss_plt.png'))

        fig2 = plt.figure()
        for key in losses:
            y = losses[key]
            y -= np.mean(y)
            y /= ( np.std(y) + 1e-6 ) 
            noPoints = len(losses[key])
            factor = float(noPoints)/(e+1)
            plt.plot(np.arange(len(losses[key]))/factor,y, label=key)
        plt.xlabel('epoch')
        plt.ylabel('normalised loss')
        plt.legend()
        fig2.savefig(join(output_path, 'norm_loss_plt.png'))

    print('full time', time() - full_time)

    sleep(5)
    done = True