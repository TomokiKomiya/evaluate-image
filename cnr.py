import cv2
import numpy as np
import math
import os
from skimage.measure import compare_ssim, compare_psnr
import time
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import os
import statistics
import math

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def average(arr):
    return statistics.mean(arr)

def std(arr):
    return statistics.stdev(arr)

def cal_cnr(mean_ori, stdev_ori, mean_dis, stdev_dis):
    return abs(mean_ori - mean_dis) / math.sqrt((pow(stdev_ori, 2) + pow(stdev_dis, 2)) / 2)

def evalute(original, distorted):
    original  = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
    distorted = cv2.imread(distorted, cv2.IMREAD_GRAYSCALE)

    print("Original Shape : ", original.shape)
    print("Distored Shape : ", distorted.shape)

    original = min_max(original)
    distorted = min_max(distorted)

    #画素値の読み込み
    pixel_value_Ori = original.flatten().astype(float)
    pixel_value_Dis = distorted.flatten().astype(float)

    #画素情報の取得
    imageHeight, imageWidth = original.shape

    #画素数
    N = imageHeight * imageWidth
    addr = N

    roi_ori_arr = []
    air_ori_arr = []
    roi_dis_arr = []
    air_dis_arr = []
    for i in range(addr):
        if pixel_value_Ori[i] > 0.15:
            roi_ori_arr.append(pixel_value_Ori[i])
        else:
            air_ori_arr.append(pixel_value_Ori[i])

        if pixel_value_Dis[i] > 0.15:
            roi_dis_arr.append(pixel_value_Dis[i])
        else:
            air_dis_arr.append(pixel_value_Dis[i])

    mean_ori = average(roi_ori_arr)
    print('平均(original): ', mean_ori)
    stdev_ori = std(air_ori_arr)
    print('標準偏差(original): ', stdev_ori)

    mean_dis = average(roi_dis_arr)
    print('平均(dis): ', mean_dis)
    stdev_dis = std(air_dis_arr)
    print('標準偏差(dis): ', stdev_dis)

    cnr_data = cal_cnr(mean_ori, stdev_ori, mean_dis, stdev_dis)
    print('CNR: ', cnr_data)

if __name__ == "__main__":
    original = os.getcwd() + '/data/cube-tetra-original_416x421_396x400.png'
    distorted = os.getcwd() + './data/cube-tetra-out_396x400.png'

    evalute(original, distorted)