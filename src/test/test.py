# pylint: disable=E0401, E0302
import sys

import argparse
from time import time
import numpy as np

from dataloader import TestData
from cvae import CVAETEST as CVAE

from torchvision import transforms
import torch.nn.functional as F

import torch
from torch.autograd import Variable
from torch import nn
from torchvision.utils import save_image

import os
from os.path import join

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

import sys

import argparse
import numpy as np
from skimage.io import imread
from PIL import Image
from PIL.ImageOps import fit
from numpy import loadtxt
from numpy import transpose
from matplotlib import pyplot as plt
import random
from time import sleep


print('pytorch version : ' + str(torch.__version__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--label', default=0, type=int)
parser.add_argument('--path', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/test_data/test.npy', type=str)
parser.add_argument('--CVAE_PATH', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/test/', type=str)
opts = parser.parse_args()

root = Tk()
root.title('Demo')
root.resizable(False, False)
windowWidth = 800
windowHeight = 500
screenWidth,screenHeight = root.maxsize()
geometryParam = '%dx%d+%d+%d'%(windowWidth, windowHeight, (screenWidth-windowWidth)/2, (screenHeight - windowHeight)/2)
root.geometry(geometryParam)
root.wm_attributes('-topmost',1)

frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH,expand=1)

def printcoords():
    File = filedialog.askopenfilename(parent=root,title='Choose an image.')

    print('changing data...')
    data_img = []
    img = imread(File)
    img = Image.fromarray(img)
    img = fit(img, size=(64, 64))
    img = transpose(img, (2, 0, 1))
    data_img.append(img)
    np.save(opts.path, np.asarray(data_img))

    test_dataset = TestData(label=opts.label, path=opts.path, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    cvae = CVAE(200).to(device)
    cvae.load_params(opts.CVAE_PATH)

    evaluation_dir= opts.CVAE_PATH + 'evalFolder'
    try:
        os.mkdir(evaluation_dir)
    except:
        print('file already created')

    cvae.eval()

    test_x, test_y = iter(dataloader).next()

    for i in range(3):
        test_rec, test_mean, test_log_var, test_predict = cvae(test_x, test_y)

        save_image(test_x.data, join(evaluation_dir,'input.png'))
        save_image(test_rec.data, join(evaluation_dir,'output_test.png'))
        x = test_x.data
        y = test_y

        mu, logVar, y = cvae.encode(x)
        z = cvae.reparameterization(mu, logVar)

        sample1 = cvae.decode(torch.LongTensor(np.ones(y.size(), dtype=int)).type_as(z), z)    
        sample2 = cvae.decode(torch.LongTensor(np.zeros(y.size(), dtype=int)).type_as(z), z)

        save_image(sample1.cpu().data, join(evaluation_dir,'sample1.png'))
        save_image(sample2.cpu().data, join(evaluation_dir,'sample2.png'))

        arr = ['input.png', 'sample1.png', 'sample2.png']
        toImage = Image.new('RGBA',(584,128))
        for j in range(3):
            fromImge = Image.open(join(evaluation_dir, arr[j]))
            fromImge = fromImge.resize((128, 128),Image.ANTIALIAS)
            loc = (128*j + 80, 0)
            toImage.paste(fromImge, loc)

        toImage.save('merged' + str(i) + '.png')

    arr = ['merged0.png', 'merged1.png', 'merged2.png']
    toImage = Image.new('RGBA',(584,384))
    for j in range(3):
        fromImge = Image.open(arr[j])
        loc = (0, 128*j)
        toImage.paste(fromImge, loc)

    toImage.save('merged.png')

    filename = ImageTk.PhotoImage(Image.open('merged.png'))
    canvas.image = filename
    canvas.create_image(124,10,anchor='nw',image=filename)

Button(root,text='choose',command=printcoords).pack()
root.mainloop()