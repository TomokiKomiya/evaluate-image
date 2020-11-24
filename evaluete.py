import cv2
import numpy as np
import math
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
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

def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    return val

def pixel_diff(img1, img2):
    return np.sum(np.absolute(img1 - img2))

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
    print(np.max(pixel_value_Dis))
    print(np.min(pixel_value_Dis))
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
    stdev_ori = std(roi_ori_arr)
    print('標準偏差(original): ', stdev_ori)

    mean_dis = average(roi_dis_arr)
    print('平均(dis): ', mean_dis)
    stdev_dis = std(roi_dis_arr)
    print('標準偏差(dis): ', stdev_dis)

    cnr_data = cal_cnr(mean_ori, stdev_ori, mean_dis, stdev_dis)
    print('CNR: ', cnr_data)

    # original  = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
    # distorted = cv2.imread(distorted, cv2.IMREAD_GRAYSCALE)

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
    #1画素あたりRGB3つの情報がある.
    addr = N
    sum = 0
    #差の2乗の総和を計算
    for i in range(addr):
        sum += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
    MSE = sum / N
    PSNR = 10 * math.log(255*255/MSE,10)
    # print('PSNR: ',PSNR)
    # print('MSE: ', MSE)

    ssim = measurement(structural_similarity, img1=original, img2=distorted)
    psnr = measurement(peak_signal_noise_ratio, img1=original, img2=distorted)

    print("ssim: ", ssim )
    print("psnr: ", psnr)
    print('CNR: ', cnr_data)

if __name__ == "__main__":
    ori = "MILLENIUMchocolate-reference-uint16_340x340x40"
    dis = "MILLENIUMchocolate-uint16_170x170x20-MILLENIUMchocolate-200image-7x7-float32_20"

    original = os.getcwd() + '/data/' + ori + '.jpg'
    distorted = os.getcwd() + './data/' + dis + '.png'

    evalute(original, distorted)