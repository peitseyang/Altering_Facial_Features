# pylint: disable=E0401, E0302
import sys
sys.path.append('../')

import argparse
from time import time
import numpy as np

from data.dataloader import TestData
from model.cvae import CVAE
from model.discriminator import Discriminator as CLASSIFIER

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

import sys

EPSILON = 1e-6

# python3 celeba_info_cVAEGAN.py --alpha 0.2 --batch_size 32 --beta 0 --delta 0.1 --fSize 32 --epochs 45 --rho 0.1

# def label_switch(x,y,cvae,exDir=None): #when y is a unit not a vector
#     print('switching label...1')
#     #get x's that have smile
#     if (y.data == 0).all(): #if no samples with label 1 use all samples
#         x0 = Variable(x)
#     else:
#         zeroIdx = torch.nonzero(y.data)
#         x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

#     #get z
#     mu, logVar, y = cvae.encode(x0)
#     z = cvae.reparameterization(mu, logVar)

#     ySmile = Variable(torch.LongTensor(np.ones(y.size(), dtype=int))).type_as(z)
#     smileSamples = cvae.decode(ySmile, z)    
    

#     yNoSmile = Variable(torch.LongTensor(np.zeros(y.size(), dtype=int))).type_as(z)
#     noSmileSamples = cvae.decode(yNoSmile, z)
    
#     if exDir is not None:
#         print('saving rec w/ and w/out label switch to', join(exDir,'rec.png'),'... ')
#         save_image(x0.data, join(exDir, 'original.png'))
#         save_image(smileSamples.cpu().data, join(exDir,'rec_1.png'))
#         save_image(noSmileSamples.cpu().data, join(exDir,'rec_0.png'))

#     return smileSamples, noSmileSamples

# def binary_class_score(pred, target, thresh=0.5):
#     predLabel = torch.gt(pred, thresh)
#     classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
#     return  classScoreTest.float().sum()/target.size(0)


print('pytorch version : ' + str(torch.__version__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--label', default=None, type=int)
parser.add_argument('--path', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/test_data/test.npy', type=str)
parser.add_argument('--load_VAE_from', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/test/', type=str)
parser.add_argument('--load_CLASSER_from', default='../../Experiments_delta_z/celeba_joint_VAE_DZ/Ex_15', type=str)
parser.add_argument('--evalMode', action='store_true')

opts = parser.parse_args()


test_dataset = TestData(label=opts.label, path=opts.path, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

cvae = CVAE(200, device).to(device)
# classer = CLASSIFIER().to(device)

cvae.load_params(opts.load_VAE_from)
# classer.load_params(opts.load_CLASSER_from)

evaluation_dir= opts.load_VAE_from + 'evalFolder'
try:  #may already have an eval folder
	os.mkdir(evaluation_dir)
except:
	print('file already created')
# _, _ = evaluate(cvae, dataloader, evaluation_dir, e='evalMode', classifier=classer)
# normbceLossTest, classScoreTest = evaluate(cvae, dataloader['test'], exDir, e=e)
# normbceLossTest, classScoreTest = evaluate(cvae, dataloader['test'], exDir, e='evalMode')

cvae.eval()

x, y = dataloader
print(x)
print(y)

# test_x, test_y = iter(test_data).next()
# test_x = Variable(test_x).to(cvae.device)
# test_y = Variable(test_y).view(test_y.size(0),1).to(cvae.device)

# z = Variable(torch.randn(test_x.size(0), opts.latent_size)).to(cvae.device)

# y_1 = Variable(torch.ones(test_y.size())).type_as(test_x)
# samples = cvae.decode(y_1, z).cpu()
# save_image(samples.data, join(exDir,'one_epoch'+str(e)+'.png'))

# y_0 = Variable(torch.zeros(test_y.size())).type_as(test_x)
# samples = cvae.decode(y_0, z).cpu()
# save_image(samples.data, join(exDir,'zero_epoch'+str(e)+'.png'))

# test_rec, test_mean, test_log_var, test_predict = cvae(test_x)


# test_bce_loss, test_kl_loss = cvae.loss(test_rec, test_x, test_mean, test_log_var)
# predict_label = torch.floor(test_predict)

# classScoreTest= binary_class_score(predict_label, test_y, thresh=0.5)
# print('classification test:', classScoreTest.data[0])

# save_image(test_x.data, join(exDir,'input.png'))
# save_image(test_rec.data, join(exDir,'output_'+str(e)+'.png'))

# rec1, rec0 = label_switch(test_x.data, test_y, cvae, exDir=exDir)

# for further eval
# if e == 'evalMode' and classer is not None:
#     classer.eval()
#     yPred0 = classer(rec0)
#     y0 = Variable(torch.LongTensor(yPred0.size()).fill_(0)).type_as(test_x)
#     class0 = binary_class_score(yPred0, y0, thresh=0.5)
#     yPred1 = classer(rec1)
#     y1 = Variable(torch.LongTensor(yPred1.size()).fill_(1)).type_as(test_x)
#     class1 = binary_class_score(yPred1, y1, thresh=0.5)

#     f = open(join(exDir, 'eval.txt'), 'w')
#     f.write('Test MSE:'+ str(F.mse_loss(outputs, xTest).data[0]))
#     f.write('Class0:'+ str(class0.data[0]))
#     f.write('Class1:'+ str(class1.data[0]))
#     f.close()


# return (test_bce_loss).data[0]/test_x.size(0), classScoreTest.data[0]