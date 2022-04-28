# ================================================== #
#   Importation of Default Module
# ================================================== #

# ================================================== #
#   Importation of 3rd Party Module
# ================================================== #
from pytictoc import TicToc
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import cv2

# ================================================== #
#   Importation of Self Development Module
# ================================================== #

# ================================================== #
#   Declaration AND Definition Of This Module Variable
# ================================================== #
clock = TicToc()

fdir = r"./../Image"
fname = r"ImageOfTesting001.bmp"
#fpath = fdir + "\\" + fname

Kernel_Prewitt_H = np.array(
    [[-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]]
)

Kernel_Prewitt_V = np.array(
    [[-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]]
)

# ================================================== #
#   Declaration AND Definition Of This Module Function
# ================================================== #

def ReadTestingImage (fdir: str=fdir, fname: str=fname) -> np.ndarray:
    fpath = fdir + "\\" + fname
    
    #imgin = img.imread(fpath)
    imgin = Image.open(fpath)
    
    return np.array(imgin)


def GenerateEdgeImage (ImgIn: np.ndarray, Kernel: np.ndarray, Mode: str="valid"):
    imgin = ImgIn.copy().astype(np.float64)
    imgout = np.zeros(imgin.shape, dtype=np.float64)
    if (Mode == "valid"):
        if (imgout.ndim == 2):
            pass
        if (imgout.ndim == 3):
            [nv, nh, nc] = imgout.shape
            [kv, kh] = Kernel.shape
            for i in np.arange(int(np.floor((kv-1)/2)), nv-((kv-1)-int(np.floor((kv-1)/2))), 1):
                for j in np.arange(int(np.floor((kh-1)/2)), nh-((kh-1)-int(np.floor((kh-1)/2))), 1):
                    for k in np.arange(nc):
                        #print(i)
                        #print(j)
                        #print(int(j-np.floor((kh-1)/2)))
                        #print(int(j-np.floor((kh-1)/2)+kh))
                        #print(imgin[int(i-np.floor((kv-1)/2)):int(i-np.floor((kv-1)/2)+kv), int(j-np.floor((kh-1)/2)):int(j-np.floor((kh-1)/2)+kh), k])
                        #print(Kernel)
                        imgout[i, j, k] = np.sum(imgin[int(i-np.floor((kv-1)/2)):int(i-np.floor((kv-1)/2)+kv), int(j-np.floor((kh-1)/2)):int(j-np.floor((kh-1)/2)+kh), k] * Kernel)
    
    return imgout
# ================================================== #
#   Declaration AND Definition Of This Module Class
# ================================================== #

# ================================================== #
#   Testing Of This Module
# ================================================== #
if (__name__ == "__main__"):
    imgin = ReadTestingImage()
    #print(imgin.shape)
    #print(imgin.dtype)
    if ((imgin.ndim == 3) and (imgin.shape[2] == 4)):
        imgin = imgin[:, :, 0:3].copy()
    #print(imgin.shape)
    #print(imgin.dtype)
    #plt.imshow(imgin)
    #plt.show()
    
    #Kernel = Kernel_Prewitt_H
    #clock.tic()
    #imgout = GenerateEdgeImage(imgin, Kernel=Kernel)
    #clock.toc()
    #plt.imshow(imgout)
    #plt.show()
    #imgout *= 255/np.max(imgout)
    #img.imsave(r"D:\Test.bmp", imgout.astype(np.uint8))
    
    # SciPy.Signal.Correlate2D
    Kernel = Kernel_Prewitt_H
    imgedge_h = np.zeros(imgin.shape, dtype=np.float64)
    clock.tic()
    if (imgin.ndim == 3):
        for k in np.arange(3):
            imgedge_h[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel, mode="same", boundary="symm")
    else:
        imgedge_h[:] = sig.correlate2d(imgin, Kernel, mode="same", boundary="symm")
    clock.toc()
    imgedge_h = np.abs(imgedge_h)
    imgedge_h *= 255/np.max(imgedge_h)
    img.imsave(r"D:\ImageEdgeH.bmp", imgedge_h.astype(np.uint8), cmap="gray")
    # Histogram
    #plt.hist(imgedge_h.flatten(), bins=np.arange(256))
    #plt.show()
    
    # SciPy.Signal.Correlate2D
    Kernel = Kernel_Prewitt_V
    imgedge_v = np.zeros(imgin.shape, dtype=np.float64)
    clock.tic()
    if (imgin.ndim == 3):
        for k in np.arange(3):
            imgedge_v[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel, mode="same", boundary="symm")
    else:
        imgedge_v[:] = sig.correlate2d(imgin, Kernel, mode="same", boundary="symm")
    clock.toc()
    imgedge_v = np.abs(imgedge_v)
    imgedge_v *= 255/np.max(imgedge_v)
    img.imsave(r"D:\ImageEdgeV.bmp", imgedge_v.astype(np.uint8), cmap="gray")
    # Histogram
    #plt.hist(imgedge_v.flatten(), bins=np.arange(256))
    #plt.show()
    
    imgedge = (imgedge_h + imgedge_v) / 2
    img.imsave(r"D:\ImageEdge.bmp", imgedge.astype(np.uint8), cmap="gray")
    
    Thres = 128
    imgedge_thres = imgedge.copy()
    imgedge_thres[imgedge_thres <= Thres] = 0
    imgedge_thres[imgedge_thres > Thres] = 255
    img.imsave(r"D:\ImageEdge_Thres{:03}.bmp".format(Thres), imgedge_thres.astype(np.uint8), cmap="gray")
    
    # Canny Using OpenCV
    Thres1 = 64
    Thres2 = 192
    imgin_cv = cv2.imread(fdir + "\\" + fname) #cv2.fromarray(imgin)
    imgedge_cv_canny = cv2.Canny(imgin_cv, Thres1, Thres2)
    imgedge_cv_canny = np.array(imgedge_cv_canny)
    img.imsave(r"D:\ImageEdge_CV_Canny_{:03}_{:03}.bmp".format(Thres1, Thres2), imgedge_cv_canny.astype(np.uint8), cmap="gray")