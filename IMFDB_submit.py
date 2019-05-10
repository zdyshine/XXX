#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.io
import os
import json
import torch.utils.data
from backbone import mobilefacenet, resnet, arcfacenet, cbam
from dataset.imfdb import IMFDB_TEST
import torchvision.transforms as transforms
from torch.nn import DataParallel
import argparse
import pandas as pd
from tqdm import tqdm


def loadModel(backbone_net, gpus='0', resume=None):

    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(args.backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(resume)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    IMFDB_testset = IMFDB_TEST(args.root, transform=transform)
    IMFDB_testloader = torch.utils.data.DataLoader(IMFDB_testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, drop_last=False)

    return net.eval(), device, IMFDB_testset, IMFDB_testloader


def getFeaturePickle(pickle_path, net, data_set, data_loader, feature_dims, batch_size):
    nums = len(data_set)
    batches = nums//batch_size
    FEATURES = np.zeros((nums, feature_dims))
    for i, input in tqdm(enumerate(data_loader)):
        input_var = torch.autograd.Variable(input, volatile=True)
        feature = net(input_var)
        feature = feature.cpu().detach().numpy()
        if i != batches-1:
            FEATURES[i*batch_size:(i+1)*batch_size, :] = feature
        elif i == batches-1:   #the last batch
            FEATURES[i*batch_size:, :] = feature
    f = open(pickle_path,"wb")
    import pickle
    pickle.dump(FEATURES, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='D:/YeJQ/IMFDB_final/AlignIMFDB/ValidationData/', help='The path of ksyun data')
    parser.add_argument('--backbone_net', type=str, default='MobileFace', help='CBAM_100_SE, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE, MobileFace')
    parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--resume', type=str, default='./model/IMFDB_MOBILEFACE_20190510_142512_Align_1.000/Iter_006000_net.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_path', type=str, default='./result/ksyun_ResIR50_feature_0415.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='0,1,2', help='gpu list')
    parser.add_argument('--pickle_path', type=str, default='D:/YeJQ/IMFDB_final/submit/submit_IMFDB_4_mobile.pkl', help='save pickle')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()

    net, device, imfdb_dataset, imfdb_loader = loadModel(args.backbone_net, args.gpus, args.resume)
    getFeaturePickle(args.pickle_path, net, imfdb_dataset, imfdb_loader, args.feature_dim, args.batch_size)
