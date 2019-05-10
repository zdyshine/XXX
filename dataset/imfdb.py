#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: casia_webface.py
@time: 2018/12/21 19:09
@desc: CASIA-WebFace dataset loader
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt



def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class IMFDB(data.Dataset):
    def __init__(self, root, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        star_list = os.listdir(root)
        starid_dict = {}

        for idx, star in enumerate(star_list):
            if star not in starid_dict.keys():
                starid_dict[star] = idx
            star_folder = os.path.join(self.root, star)
            file_list = os.listdir(star_folder)
            for image_name in file_list:
                file_path = os.path.join(star_folder, image_name)
                image_list.append(file_path)
                label_list.append(idx)

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)


    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(img_path)

        img = cv2.resize(img, (112, 112))

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return len(self.image_list)


class IMFDB_TEST(data.Dataset):
    def __init__(self, root, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        test_image_list = []
        test_list = os.listdir(root)

        for idx, test_image in enumerate(test_list):
            test_image_list.append(os.path.join(root, test_image))

        self.test_image_list = test_image_list


    def __getitem__(self, index):
        test_path = self.test_image_list[index]

        img = self.loader(test_path)

        img = cv2.resize(img, (112, 112))

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img

    def __len__(self):
        return len(self.test_image_list)


if __name__ == '__main__':
    root = 'D:/YeJQ/IMFDB_final_rename/'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = IMFDB(root, transform=transform)
    #import pdb; pdb.set_trace()
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False) #batch_size must be divided by len(self.image_list).
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
        img1 = transforms.ToPILImage()(data[0][0])
        img1.show()
        print(data[1])
        import pdb; pdb.set_trace()
