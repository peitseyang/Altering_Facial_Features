# pylint: disable=E0401
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

import itertools
import threading
import time
import sys

done = False
# here is the animation

def animate():
    for c in itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠴', '⠦', '⠧', '⠇', '⠏']):
        if done:
            break
        sys.stdout.write('\r' + c + ' ')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

t = threading.Thread(target=animate)
t.start()

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/raw_data/', type=str)
parser.add_argument('--test', dest='test_mode', action='store_true')
parser.add_argument('--make', dest='test_mode', action='store_false')
parser.set_defaults(test_mode=False)
parser.add_argument('--attr', default='Smiling', type=str)
opts = parser.parse_args()
print(opts.test_mode)

attributes = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young" 
]

def make_data():
	print('making data...')

	img_path = opts.root + 'img_align_celeba/'
	attr_path = opts.root + 'list_attr_celeba.txt'

	f = open(attr_path)
	f.readline()
	labels_name = f.readline().split(' ')
	data_img = []
	data_label = []
	attr_index = attributes.index(opts.attr)
	count_true = 0
	count_false = 0

	for index, line in enumerate(f):
		line_array = line.split(' ')
		img_name = line_array[0]
		labels = line_array[1:]

		img = imread(img_path + img_name)
		img = Image.fromarray(img)
		img = fit(img, size=(64, 64)) # (64, 64, 3)
		label = loadtxt(labels).astype('int')
		img = transpose(img, (2, 0, 1)) # (3, 64, 64)
		data_img.append(img)
		data_label.append(label)
		if label[attr_index] == 1:
			count_true += 1
		else:
			count_false += 1

		if opts.test_mode and index is 100:
			break

	print(np.shape(data_img))
	print(np.shape(data_label))

	if not opts.test_mode:
		print('true: ' + str(count_true))
		print('false: ' + str(count_false))
		np.save('./celebA/img.npy', np.asarray(data_img))
		np.save('./celebA/attr.npy', np.asarray(data_label))
	else:
		print('test_mode')
		true_label = 0
		false_label = 0
		while true_label is 0 or false_label is 0:
			# print(true_label)
			# print(false_label)
			index = random.randint(0,100)
			label = data_label[index][attr_index]
			if label == 1 and true_label == 0:
				plt.figure()
				plt.title('true')
				true_label = 1
				plt.imshow(data_img[index].transpose(1,2,0))
			elif label == -1 and false_label == 0:
				plt.figure()
				plt.title('false')
				false_label = 1
				plt.imshow(data_img[index].transpose(1,2,0))
			# data_label = np.asarray(data_label)
			# data_img = np.asarray(data_img)
		
		plt.show()

make_data()
done = True








