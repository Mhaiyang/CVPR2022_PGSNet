"""
 @Time    : 2021/10/10 16:20
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : cvpr2022
 @File    : misc.py
 @Function:
 
"""
import os
import xlwt
import numpy as np
from PIL import Image

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="sheet1", cell_overwrite_ok=True)

    j = 0
    for data in datas:
        for i in range(len(data)):
            sheet1.write(i, j, data[i])
        j = j + 1

    f.save(file_path)

def reorder(image):
    x, y, z = image.split()
    new_image = Image.merge("RGB", (z, y, x))

    return new_image

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_ber(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 100 * (1 - (1 / 2) * ((TP / N_p) + (TN / N_n)))

    return ber_

def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
