# pylint: disable=E0401
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from random import randint
import sys


path = '/Users/alexyang/Desktop/final_project/Altering_Facial_Features/src/data/celebA/'
# path = '/home/csie-owob/alexyang/sinb/cvae_gan/src/data/celebA/'

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

class CelebA(data.Dataset):
    def __init__(self, label, path, train=True, transform=None):
        self.data_path = path + 'img.npy'
        self.label_path = path + 'attr.npy'
        self.train = train
        self.transform = transform
        self.attribute_index = attributes.index(label)
        
        if self.train:
            self.train_data = np.load(self.data_path)[100:]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            train_labels = np.load(self.label_path)[100:, self.attribute_index]
            self.train_labels = (train_labels.astype(int) + 1) / 2
        else:
            self.test_data = np.load(self.data_path)[:100]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
            test_labels = np.load(self.label_path)[:100, self.attribute_index]
            self.test_labels = (test_labels.astype(int) + 1) / 2

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.test_data)
        
    def __getitem__(self, index, UnitTest=False):
        if self.train:
            img, label = Image.fromarray(self.train_data[index]), self.train_labels[index]
        else:
            img, label = Image.fromarray(self.test_data[index]), self.test_labels[index]
        if self.transform is not None and UnitTest is False:
            img = self.transform(img)
        label = label.astype(int)
        return img, label


def UnitTest(label):
    print('Unit testing class CelebA with ' + label + ' attribute')
    celebA_data = CelebA(label, path, train=False, transform=transforms.ToTensor())
    img, label = celebA_data.__getitem__(randint(0, celebA_data.__len__()), UnitTest=True)
    print(np.shape(img))
    img.save(str(label) + '.png')
    img.show()


try:
    if len(sys.argv) == 2 and attributes.index(sys.argv[1]) > 0:
        UnitTest(sys.argv[1])
    else:
        UnitTest('Smiling')
except ValueError:
    print('Attribute is not in the list')











