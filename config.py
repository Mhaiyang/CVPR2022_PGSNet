"""
 @Time    : 2021/10/16 10:36
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : cvpr2022
 @File    : config.py
 @Function:
 
"""
import os

backbone_path1 = './backbone/Conformer_base_patch16.pth'
backbone_path2 = './backbone/Conformer_base_patch16.pth'
backbone_path3 = './backbone/Conformer_base_patch16.pth'
backbone_path4 = './backbone/resnet/resnet18-5c106cde.pth'

datasets_root = './data/RGBP-Glass'

training_root = os.path.join(datasets_root, 'train')
testing_root = os.path.join(datasets_root, 'test')

pgsnet_ckpt_path = './ckpt/PGSNet.pth'
