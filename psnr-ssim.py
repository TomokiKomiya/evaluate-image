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

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    return val

def pixel_diff(img1, img2):
    return np.sum(np.absolute(img1 - img2))

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
    #1画素あたりRGB3つの情報がある.
    addr = N
    sum = 0
    #差の2乗の総和を計算
    for i in range(addr):
        sum += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
    MSE = sum / N
    PSNR = 10 * math.log(255*255/MSE,10)
    print('PSNR: ',PSNR)
    print('MSE: ', MSE)

    ssim = measurement(compare_ssim, img1=original, img2=distorted)
    psnr = measurement(compare_psnr, img1=original, img2=distorted)

    print("ssim: ", ssim )
    print("psnr: ", psnr)

    print("PSNR(CV2): ", cv2.PSNR(original, distorted))

if __name__ == "__main__":
    original = os.getcwd() + '/data/cube-tetra-reference-uint16_64.png'
    distorted = os.getcwd() + './data/cube-tetra-uint16_200x200x50-cube-tetra-200image-13x13-float32_64.png'

    evalute(original, distorted)
