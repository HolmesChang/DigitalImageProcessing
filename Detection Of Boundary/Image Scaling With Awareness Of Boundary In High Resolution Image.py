import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

Ratio = 3/2

Kernel_Sobel_H = np.array(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
)

Kernel_Sobel_V = np.array(
    [[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
)

GradientThreshold = 32

imgin = img.imread(r"Image\ImageOfTesting004.bmp")
(nvi, nhi, nci) = imgin.shape

imgout_BiLinear = cv2.resize(imgin, dsize=(int(nhi*Ratio), int(nvi*Ratio)), interpolation=cv2.INTER_LINEAR)
img.imsave(r"Imgout_BiLinear.bmp", imgout_BiLinear)

imgout_Sobel_H = np.zeros(imgout.shape, dtype=np.float64)
imgout_Sobel_V = np.zeros(imgout.shape, dtype=np.float64)
for k in np.arange(nci):
    imgout_Sobel_H[:, :, k] = sig.correlate2d(imgout_BiLinear[:, :, k], Kernel_Sobel_H, mode="same", boundary="symm")
    imgout_Sobel_V[:, :, k] = sig.correlate2d(imgout_BiLinear[:, :, k], Kernel_Sobel_V, mode="same", boundary="symm")

imgout_Sobel = (imgout_Sobel_H**2 + imgout_Sobel_V**2)**0.5
img.imsave(r"Imgout_Sobel_From_High_Resolution_Image.bmp", (imgout_Sobel*255/np.max(imgout_Sobel)).astype(np.uint8))
