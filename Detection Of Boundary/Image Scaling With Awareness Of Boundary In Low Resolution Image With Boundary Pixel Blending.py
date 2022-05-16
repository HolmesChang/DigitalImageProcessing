from pytictoc import TicToc
import numpy as np
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

#mpl.use("TkAgg")

clock = TicToc()

clock.tic()

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

#Kernel_Sobel_H = np.array(
#    [[-1, 1],
#    [-2, 2],
#    [-1, 1]]
#)
#
#Kernel_Sobel_V = np.array(
#    [[-1, -2, -1],
#    [1, 2, 1]]
#)

#Kernel_Sobel_H = np.array(
#    [[0, 0],
#    [-1, 1],
#    [0, 0]]
#)
#
#Kernel_Sobel_V = np.array(
#    [[0, -1, 0],
#    [0, 1, 0]]
#)

#Kernel_Sobel_H = np.array(
#    [[0, 0, 0],
#    [-1, 0, 1],
#    [0, 0, 0]]
#)
#
#Kernel_Sobel_V = np.array(
#    [[0, -1, 0],
#    [0, 0, 0],
#    [0, 1, 0]]
#)

GradientThreshold = 32

imgin = img.imread(r"Image\ImageOfTesting003.bmp")
(nvi, nhi, nci) = imgin.shape

imgin_Sobel_H = np.zeros(imgin.shape, dtype=np.float64)
imgin_Sobel_V = np.zeros(imgin.shape, dtype=np.float64)
for k in np.arange(nci):
    imgin_Sobel_H[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel_Sobel_H, mode="same", boundary="symm")
    imgin_Sobel_V[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel_Sobel_V, mode="same", boundary="symm")

imgin_Sobel = (imgin_Sobel_H**2 + imgin_Sobel_V**2)**0.5
img.imsave(r"Imgin_Sobel.bmp", (imgin_Sobel*255/np.max(imgin_Sobel)).astype(np.uint8))

imgout_NN = cv2.resize(imgin, dsize=(int(nhi*Ratio), int(nvi*Ratio)), interpolation=cv2.INTER_NEAREST)
imgout_Sobel_H = np.zeros((int(nvi*Ratio), int(nhi*Ratio), nci), dtype=np.float64)
imgout_Sobel_V = np.zeros((int(nvi*Ratio), int(nhi*Ratio), nci), dtype=np.float64)
imgout_Sobel = np.zeros((int(nvi*Ratio), int(nhi*Ratio), nci), dtype=np.float64)
imgout = np.zeros((int(nvi*Ratio), int(nhi*Ratio), nci), dtype=np.uint8)
(nvo, nho, nco) = imgout.shape
for i in np.arange(nvi):
    for j in np.arange(nhi):
        for k in np.arange(nci):
            if (imgin_Sobel[i, j, k] > GradientThreshold):
                #imgout_Sobel_H[int(np.round(i*Ratio)), int(np.round(j*Ratio)), k] = imgin_Sobel_H[i, j, k]
                #imgout_Sobel_V[int(np.round(i*Ratio)), int(np.round(j*Ratio)), k] = imgin_Sobel_V[i, j, k]
                #imgout_Sobel[int(np.round(i*Ratio)), int(np.round(j*Ratio)), k] = imgin_Sobel[i, j, k]
                #imgout[int(np.round(i*Ratio)), int(np.round(j*Ratio)), k] = imgin[i, j, k]
                
                imgout_Sobel_H[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel_H[i, j, k]
                imgout_Sobel_V[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel_V[i, j, k]
                imgout_Sobel[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel[i, j, k]
                imgout[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin[i, j, k]
                
                #imgout_Sobel_H[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel_H[i, j, k]
                #imgout_Sobel_V[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel_V[i, j, k]
                #imgout_Sobel[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgin_Sobel[i, j, k]
                #imgout[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k] = imgout_NN[int(np.floor(i*Ratio)), int(np.floor(j*Ratio)), k]

img.imsave(r"Imgout_Sobel_01.bmp", (imgout_Sobel*255/np.max(imgout_Sobel)).astype(np.uint8))
img.imsave(r"Imgout_BoundaryPixel.bmp", imgout)
img.imsave(r"Imgout_NN.bmp", imgout_NN)

imgout_BiLinear = cv2.resize(imgin, dsize=(int(nhi*Ratio), int(nvi*Ratio)), interpolation=cv2.INTER_LINEAR)
#for i in np.arange(nvo):
#    for j in np.arange(nho):
#        for k in np.arange(nco):
#            if ((i < (nvo-1)) and (j < (nho-1))):
#                if (np.sum(imgout_Sobel[(i-1):(i+2), (j-1):(j+2), k]) == 0):
#                    imgout[i, j, k] = imgout_BiLinear[i, j, k]
for i in np.arange(nvo):
    for j in np.arange(nho):
        tmp_i = i / Ratio
        tmp_j = j / Ratio
        x = tmp_j % 1
        y = tmp_i % 1
        for k in np.arange(nco):
            if (imgout[i, j, k] == 0):
                if ((i < (nvo-1)) and (j < (nho-1))):
                    if (np.sum(imgin_Sobel[int(tmp_i):(int(tmp_i)+2), int(tmp_j):(int(tmp_j)+2), k] <= GradientThreshold) == 4):
                        imgout[i, j, k] = imgin[int(tmp_i), int(tmp_j), k].astype(np.float64)       * (1-x)     * (1-y) +\
                                          imgin[int(tmp_i), int(tmp_j)+1, k].astype(np.float64)     * x         * (1-y) +\
                                          imgin[int(tmp_i)+1, int(tmp_j), k].astype(np.float64)     * (1-x)     * y +\
                                          imgin[int(tmp_i)+1, int(tmp_j)+1, k].astype(np.float64)   * x         * y

img.imsave(r"Imgout_BiLinear.bmp", imgout_BiLinear)
img.imsave(r"Imgout_BoundaryPixel+BackgroundPixel.bmp", imgout)

# Horizontal
for i in np.arange(nvo):
    for j in np.arange(nho):
        for k in np.arange(nco):
            if ((i > 0) and (i < (nvo-1)) and (j < (nho-1))):
                if ((imgout_Sobel[i-1, j, k] > 0) and (imgout_Sobel[i+1, j, k] > 0) and (imgout_Sobel_V[i-1, j, k] == 0) and (imgout_Sobel_V[i+1, j, k] == 0)):
                    imgout[i, j, k] = np.mean((imgout[i-1, j, k], imgout[i+1, j, k]))
                    imgout_Sobel_H[i, j, k] = np.mean((imgout_Sobel_H[i-1, j, k], imgout_Sobel_H[i+1, j, k]))
                    imgout_Sobel_V[i, j, k] = np.mean((imgout_Sobel_V[i-1, j, k], imgout_Sobel_V[i+1, j, k]))
                    imgout_Sobel[i, j, k] = (imgout_Sobel_H[i, j, k]**2 + imgout_Sobel_V[i, j, k]**2)**0.5

img.imsave(r"Imgout_BoundaryPixel+BackgroundPixel+VBoundary.bmp", imgout)
img.imsave(r"Imgout_Sobel_02.bmp", (imgout_Sobel*255/np.max(imgout_Sobel)).astype(np.uint8))

# Vertical
for i in np.arange(nvo):
    for j in np.arange(nho):
        for k in np.arange(nco):
            if ((i > 0) and (i < (nvo-1)) and (j < (nho-1))):
                if ((imgout_Sobel[i, j-1, k] > 0) and (imgout_Sobel[i, j+1, k] > 0) and (imgout_Sobel_H[i, j-1, k] == 0) and (imgout_Sobel_H[i, j+1, k] == 0)):
                    imgout[i, j, k] = np.mean((imgout[i, j-1, k], imgout[i, j+1, k]))
                    imgout_Sobel_H[i, j, k] = np.mean((imgout_Sobel_H[i, j-1, k], imgout_Sobel_H[i, j+1, k]))
                    imgout_Sobel_V[i, j, k] = np.mean((imgout_Sobel_V[i, j-1, k], imgout_Sobel_V[i, j+1, k]))
                    imgout_Sobel[i, j, k] = (imgout_Sobel_H[i, j, k]**2 + imgout_Sobel_V[i, j, k]**2)**0.5

img.imsave(r"Imgout_BoundaryPixel+BackgroundPixel+VBoundary+HBoundary.bmp", imgout)
img.imsave(r"Imgout_Sobel_03.bmp", (imgout_Sobel*255/np.max(imgout_Sobel)).astype(np.uint8))

# Non Boundary Pixel With Boundary Pixel Neighbor
for i in np.arange(nvo):
    for j in np.arange(nho):
        tmp_i = i / Ratio
        tmp_j = j / Ratio
        x = tmp_j % 1
        y = tmp_i % 1
        for k in np.arange(nco):
            if (imgout_Sobel[i, j, k] == 0):
                #imgout[i, j, k] = imgout_NN[i, j, k]
                imgout[i, j, k] = imgout_BiLinear[i, j, k]
                #imgout[i, j, k] = np.mean((imgout_NN[i, j, k].astype(np.float64), imgout_BiLinear[i, j, k].astype(np.float64))) 

img.imsave(r"Imgout_Finally.bmp", imgout)

clock.toc()
