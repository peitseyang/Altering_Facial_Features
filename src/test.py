# pylint: disable=E0401, E0302

import argparse
from time import time
import numpy as np

from data.dataloader import CelebA
from model.cvae import CVAE
from model.discriminator import Discriminator
from model.aux import Aux
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

EPSILON = 1e-6

# python3 celeba_info_cVAEGAN.py --alpha 0.2 --batch_size 32 --beta 0 --delta 0.1 --fSize 32 --epochs 45 --rho 0.1

def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size(0),1).type_as(x)
	return x,y

def label_switch(x,y,cvae,exDir=None): #when y is a unit not a vector
    print('switching label...1')
    #get x's that have smile
    print(type(x))
    if (y.data == 0).all(): #if no samples with label 1 use all samples
        x0 = Variable(x)
    else:
        zeroIdx = torch.nonzero(y.data)
        x0 = Variable(torch.index_select(x, dim=0, index=zeroIdx[:,0])).type_as(x)

    #get z
    mu, logVar, y = cvae.encode(x0)
    z = cvae.reparameterization(mu, logVar)

    ySmile = Variable(torch.LongTensor(np.ones(y.size(), dtype=int))).type_as(z)
    print(np.shape(z))
    print(np.shape(ySmile))
    smileSamples = cvae.decode(ySmile, z)    
    

    yNoSmile = Variable(torch.LongTensor(np.zeros(y.size(), dtype=int))).type_as(z)
    noSmileSamples = cvae.decode(yNoSmile, z)
    
    if exDir is not None:
        print('saving rec w/ and w/out label switch to', join(exDir,'rec.png'),'... ')
        save_image(x0.data, join(exDir, 'original.png'))
        save_image(smileSamples.cpu().data, join(exDir,'rec_1.png'))
        save_image(noSmileSamples.cpu().data, join(exDir,'rec_0.png'))

    return smileSamples, noSmileSamples

def binary_class_score(pred, target, thresh=0.5):
    predLabel = torch.gt(pred, thresh)
    classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
    return  classScoreTest.float().sum()/target.size(0)

def evaluate(cvae, testLoader, exDir, e=1, classifier=None):  #e is the epoch

	cvae.eval()

	#Load test data
	xTest, yTest = prep_data(iter(testLoader).next(), True)
	

	print('saving a set of samples')
	if True:
		z = Variable(torch.randn(xTest.size(0), opts.latent_size).cuda())
	else:
		z = Variable(torch.randn(xTest.size(0), opts.latent_size))

	ySmile = Variable(torch.Tensor(yTest.size()).fill_(1)).type_as(yTest)
	samples = cvae.decode(ySmile, z).cpu()
	save_image(samples.data, join(exDir,'smile_epoch'+str(e)+'.png'))

	yNoSmile = Variable(torch.Tensor(yTest.size()).fill_(0)).type_as(yTest)
	samples = cvae.decode(yNoSmile, z).cpu()
	save_image(samples.data, join(exDir,'no_smile_epoch'+str(e)+'.png'))

	#check reconstructions after each 10 epochs
	outputs, outMu, outLogVar, outY = cvae(xTest)


	bceLossTest, klLossTest = cvae.loss(outputs, xTest, outMu, outLogVar)
	predLabel = torch.floor(outY)

	classScoreTest= binary_class_score(predLabel, yTest, thresh=0.5)
	print('classification test:', classScoreTest.data[0])

	save_image(xTest.data, join(exDir,'input.png'))
	save_image(outputs.data, join(exDir,'output_'+str(e)+'.png'))

	rec1, rec0 = label_switch(xTest.data, yTest, cvae, exDir=exDir)

	# for further eval
	if e == 'evalMode' and classer is not None:
		classer.eval()
		yPred0 = classer(rec0)
		y0 = Variable(torch.LongTensor(yPred0.size()).fill_(0)).type_as(yTest)
		class0 = binary_class_score(yPred0, y0, thresh=0.5)
		yPred1 = classer(rec1)
		y1 = Variable(torch.LongTensor(yPred1.size()).fill_(1)).type_as(yTest)
		class1 = binary_class_score(yPred1, y1, thresh=0.5)

		f = open(join(exDir, 'eval.txt'), 'w')
		f.write('Test MSE:'+ str(F.mse_loss(outputs, xTest).data[0]))
		f.write('Class0:'+ str(class0.data[0]))
		f.write('Class1:'+ str(class1.data[0]))
		f.close()


	return (bceLossTest).data[0]/xTest.size(0), classScoreTest.data[0]

def make_new_folder(exDir):
    if os.path.isdir(exDir) is False:
        os.mkdir(exDir)
    i=1
    while os.path.isdir(join(exDir,'Ex_'+str(i))):
        i+=1

    os.mkdir(join(exDir,'Ex_'+str(i)))
    return join(exDir,'Ex_'+str(i))

def sample_z(batch_size, nz, useCUDA):
    if useCUDA:
        return Variable(torch.randn(batch_size, nz).cuda())
    else:
        return Variable(torch.randn(batch_size, nz))

if __name__=='__main__':

	print('pytorch version : ' + str(torch.__version__))

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--label', default='Smiling', type=str)
	parser.add_argument('--path', default='/home/csie-owob/alexyang/yerin/Altering_Facial_Features/src/data/celebA/', type=str)
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--latent_size', default=200, type=int)

	parser.add_argument('--lr', default=0.0002, type=float)
	parser.add_argument('--momentum', default=0.5, type=float)
	parser.add_argument('--weight_decay', default=0.01, type=float)

	parser.add_argument('--epochs', default=10, type=int)

	parser.add_argument('--alpha', default=1, type=float, help='p1') #weight on the KL divergance
	parser.add_argument('--rho', default=1, type=float, help='p2') #weight on the class loss for the vae
	parser.add_argument('--beta', default=1, type=float, help='p3')  #weight on the rec class loss to update VAE
	parser.add_argument('--gamma', default=1, type=float, help='p4') #weight on the aux enc loss
	parser.add_argument('--delta', default=1, type=float, help='p5') #weight on the adversarial loss
	
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	parser.add_argument('--outDir', default='./ex', type=str)
	parser.add_argument('--commit', default='None', type=str)
	parser.add_argument('--comments', type=str)
	parser.add_argument('--load_VAE_from', default=None, type=str)
	parser.add_argument('--load_CLASSER_from', default='../../Experiments_delta_z/celeba_joint_VAE_DZ/Ex_15', type=str)
	parser.add_argument('--evalMode', action='store_true')
	

	opts = parser.parse_args()


	train_dataset = CelebA(label=opts.label, path=opts.path, transform=transforms.ToTensor())
	test_dataset = CelebA(label=opts.label, path=opts.path, train=False, transform=transforms.ToTensor())
	dataloader = {
		'train': torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True),
		'test': torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
	}


	cvae = CVAE(opts.latent_size, device).to(device)
	dis = Discriminator().to(device)
	aux = Aux(opts.latent_size).to(device)
	classer = CLASSIFIER().to(device) #for eval only! 


	# #load model is applicable
	# if opts.load_VAE_from is not None:
	# 	cvae.load_params(opts.load_VAE_from)

	# if opts.evalMode:
	# 	classer.load_params(opts.load_CLASSER_from)

	# 	assert opts.load_VAE_from is not None
	# 	#make a new folder to save eval results w/out affecting others
	# 	evalDir=join(opts.load_VAE_from,'evalFolder')
	# 	print('Eval results will be saved to', evalDir)
	# 	try:  #may already have an eval folder
	# 		os.mkdir(evalDir)
	# 	except:
	# 		print('file already created')
	# 	_, _ = evaluate(cvae, testLoader, evalDir, e='evalMode', classifier=classer)
	# 	exit()


	print(cvae)
	print(dis)
	print(aux)


	optimizer_cvae = torch.optim.RMSprop(cvae.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
	optimizer_dis = torch.optim.RMSprop(dis.parameters(), lr=opts.lr, alpha=opts.momentum, weight_decay=opts.weight_decay)
	optimizer_aux = torch.optim.RMSprop(aux.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print('Outputs will be saved to:',exDir)
	# save_input_args(exDir, opts)  #save training opts


	# losses = {'total':[], 'kl':[], 'bce':[], 'dis':[], 'gen':[], 'test_bce':[], 'class':[], 'test_class':[], 'aux':[], 'auxEnc':[]}
	# Ns = len(dataloader['train'])*opts.batch_size  #no samples
	# Nb = len(dataloader['train'])  #no batches


	s_full_time = time()

	for e in range(opts.epochs):
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
			print(y.type())


			rec, mean, log_var, predict = cvae(x)
			z = cvae.reparameterization(mean, log_var)
			rec_loss, kl_loss = cvae.loss(rec, x, mean, log_var)
			en_de_coder_loss = rec_loss + opts.alpha * kl_loss

			loss = nn.BCELoss()
			class_loss = loss(predict.type_as(x), y.type_as(x))
			en_de_coder_loss += opts.rho * class_loss

			rec_mean, rec_log_var, rec_predict = cvae.encode(rec)
			rec_class_loss = loss(rec_predict.type_as(x), y.type_as(x))
			en_de_coder_loss += opts.beta * rec_class_loss

			auxY = aux(z)
			aux_en_loss = loss(auxY.type_as(x), y.type_as(x))  
			en_de_coder_loss -= opts.gamma * aux_en_loss

			auxY = aux(z.detach())  #detach: to ONLY update the AUX net #the prediction here for GT being predY
			aux_loss = loss(auxY.type_as(x), y.type_as(x)) #correct order  #predY is a Nx2 use 2nd col.
			
			dis_real = dis(x)
			dis_fake_rec = dis(rec.detach())
			randn_z = sample_z(x.size(0), opts.latent_size, True)
			randn_y = y.type_as(x)
			dis_fake_randn = dis(cvae.decode(randn_y, randn_z).detach())
			label_fake = Variable(torch.Tensor(dis_real.size()).zero_()).type_as(dis_real)
			label_real = Variable(torch.Tensor(dis_real.size()).fill_(1)).type_as(dis_real)
			loss = nn.BCELoss(size_average=False)
			dis_loss = 0.3 * (loss(dis_real, label_real) + loss(dis_fake_rec, label_fake) + loss(dis_fake_randn, label_fake)) / dis_real.size(1)

			dis_fake_rec = dis(rec)
			dis_fake_randn = dis(cvae.decode(randn_y, randn_z))
			gen_loss = 0.5 * (loss(dis_fake_rec, label_real) + loss(dis_fake_randn, label_real)) / dis_fake_rec.size(1)
			en_de_coder_loss += opts.delta * gen_loss


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
			e_kl_loss += kl_loss.item()
			e_rec_loss += rec_loss.item()
			e_gen_loss += gen_loss.item()
			e_dis_loss += dis_loss.item()
			e_class_loss += class_loss.item()
			e_aux_loss += aux_loss.item()
			e_aux_en_loss += aux_en_loss.item()

			if i%100==1:
				i+=1
				print('[%d, %d] loss: %0.5f, gen: %0.5f, dis: %0.5f, bce: %0.5f, kl: %0.5f, aux: %0.5f, time: %0.3f' % \
		 			(e, i, e_loss/i, e_dis_loss/i, e_gen_loss/i, e_rec_loss/i, e_kl_loss/i, e_aux_loss/i, time() - s_epoch_time))

	
		#generate samples after each 10 epochs

		normbceLossTest, classScoreTest = evaluate(cvae, dataloader['test'], exDir, e=e)

		# cvae.save_params(exDir=exDir)

		# losses['total'].append(e_loss/Ns)
		# losses['kl'].append(e_kl_loss/Ns)
		# losses['bce'].append(e_rec_loss/Ns)
		# losses['test_bce'].append(normbceLossTest)
		# losses['dis'].append(e_dis_loss/Ns)
		# losses['gen'].append(e_gen_loss/Ns)
		# losses['class'].append(e_class_loss/Ns)
		# losses['test_class'].append(classScoreTest)
		# losses['aux'].append(e_aux_loss/Ns)
		# losses['auxEnc'].append(e_aux_en_loss/Ns)

		# if e > 1:
		# 	plot_losses(losses, exDir, epochs=e+1)
		# 	plot_norm_losses(losses, exDir, epochs=e+1)

	normbceLossTest, classScoreTest = evaluate(cvae, dataloader['test'], exDir, e='evalMode')


