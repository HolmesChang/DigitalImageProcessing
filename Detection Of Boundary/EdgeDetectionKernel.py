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
fname = r"ImageOfTesting004.bmp"
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
    
    if ((imgin.ndim == 3) and (imgin.shape[2] == 4)):
        imgin = imgin[:, :, 0:3].copy()
    
    # ROI
    (VU, VD, HL, HR) = (0, 1000, 0, 1000)
    if (imgin.ndim == 3):
        imgin = imgin[VU:VD, HL:HR, :].copy()
    if (imgin.ndim == 2):
        imgin = imgin[VU:VD, HL:HR].copy()
    
    # Self Development Boundary Detection
    #Kernel = Kernel_Prewitt_H
    #clock.tic()
    #imgout = GenerateEdgeImage(imgin, Kernel=Kernel)
    #clock.toc()
    #plt.imshow(imgout)
    #plt.show()
    #imgout *= 255/np.max(imgout)
    #img.imsave(r"D:\Test.bmp", imgout.astype(np.uint8))
    
    # SciPy.Signal.Correlate2D
    #Kernel = Kernel_Prewitt_H
    #imgedge_h = np.zeros(imgin.shape, dtype=np.float64)
    #clock.tic()
    #if (imgin.ndim == 3):
    #    for k in np.arange(3):
    #        imgedge_h[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel, mode="same", boundary="symm")
    #else:
    #    imgedge_h[:] = sig.correlate2d(imgin, Kernel, mode="same", boundary="symm")
    #clock.toc()
    #imgedge_h = np.abs(imgedge_h)
    #imgedge_h *= 255/np.max(imgedge_h)
    #img.imsave(r"D:\ImageEdgeH.bmp", imgedge_h.astype(np.uint8), cmap="gray")
    # Histogram
    #plt.hist(imgedge_h.flatten(), bins=np.arange(256))
    #plt.show()
    
    # SciPy.Signal.Correlate2D
    #Kernel = Kernel_Prewitt_V
    #imgedge_v = np.zeros(imgin.shape, dtype=np.float64)
    #clock.tic()
    #if (imgin.ndim == 3):
    #    for k in np.arange(3):
    #        imgedge_v[:, :, k] = sig.correlate2d(imgin[:, :, k], Kernel, mode="same", boundary="symm")
    #else:
    #    imgedge_v[:] = sig.correlate2d(imgin, Kernel, mode="same", boundary="symm")
    #clock.toc()
    #imgedge_v = np.abs(imgedge_v)
    #imgedge_v *= 255/np.max(imgedge_v)
    #img.imsave(r"D:\ImageEdgeV.bmp", imgedge_v.astype(np.uint8), cmap="gray")
    # Histogram
    #plt.hist(imgedge_v.flatten(), bins=np.arange(256))
    #plt.show()
    #
    #imgedge = (imgedge_h + imgedge_v) / 2
    #img.imsave(r"D:\ImageEdge.bmp", imgedge.astype(np.uint8), cmap="gray")
    #
    #Thres = 128
    #imgedge_thres = imgedge.copy()
    #imgedge_thres[imgedge_thres <= Thres] = 0
    #imgedge_thres[imgedge_thres > Thres] = 255
    #img.imsave(r"D:\ImageEdge_Thres{:03}.bmp".format(Thres), imgedge_thres.astype(np.uint8), cmap="gray")
    
    # Image Inputting Using OpenCV
    imgin_cv = cv2.imread(fdir + "\\" + fname) #cv2.fromarray(imgin)
    
    (VU, VD, HL, HR) = (0, 0, 0, 0)
    if (VD):
        imgin_cv = imgin_cv[VU:VD, HL:HR, :].copy()
    
    (nvi, nhi, nci) = imgin_cv.shape
    
    # BiLinear Image Up Scaling Using OpenCV
    Ratio = 2
    imgout_cv = cv2.resize(imgin_cv, dsize=(nhi*Ratio, nvi*Ratio), interpolation=cv2.INTER_LINEAR)
    img.imsave(r"D:\ImageScaling_CV_{}x.bmp".format(Ratio), imgout_cv[:, :, [2, 1, 0]].astype(np.uint8))
    (nvo, nho, nco) = imgout_cv.shape
    
    # Canny Using OpenCV
    Thres1 = 64
    Thres2 = 192
    
    imgedge_cv_canny_H = np.zeros(imgout_cv.shape, dtype=np.uint8)
    for k in np.arange(nco):
        imgedge_cv_canny_H[:, :, k] = cv2.Canny(imgout_cv[:, :, :], Thres1, Thres2)
    #img.imsave(r"D:\ImageEdge_CV_Canny_H_{:03}_{:03}.bmp".format(Thres1, Thres2), imgedge_cv_canny_H, cmap="gray")
    img.imsave(r"D:\ImageEdge_CV_Canny_H_{:03}_{:03}.bmp".format(Thres1, Thres2), imgedge_cv_canny_H)
    
    #imgedge_cv_canny_L = cv2.resize(imgedge_cv_canny_H, dsize=(nvi, nhi), interpolation=cv2.INTER_NEAREST)
    imgedge_cv_canny_L = np.zeros((nvi, nhi, nci), dtype=np.uint8)
    
    for i in np.arange(nvi):
        for j in np.arange(nhi):
            for k in np.arange(nci):
                if (np.sum(imgedge_cv_canny_H[2*i:2*i+2, 2*j:2*j+2, k])):
                    imgedge_cv_canny_L[i, j, k] = 255
    
    img.imsave(r"D:\ImageEdge_CV_Canny_L.bmp", imgedge_cv_canny_L.astype(np.uint8), cmap="gray")
    
    
    # HES Only For 2x
    Statistics = np.zeros((5, 1))
    imgout = np.zeros((nvo, nho, nco), dtype=np.uint8)
    print("HES: ", end="")
    clock.tic()
    for i in np.arange(1, nvo-1, 1):
        for j in np.arange(1, nho-1, 1):
            ## Edge Pixel
            #if (imgedge_cv_canny_H[i, j]):
            #    imgout[i, j, :] = imgin_cv[int(i/2), int(j/2), :]
            #    Statistics[0, 0] += 1
            #    continue
            
            # Non Edge Pixel
            #print("i=", i, ", j=", j)
            for k in np.arange(nco):
                # Edge Pixel
                if (imgedge_cv_canny_H[i, j, k]):
                    imgout[i, j, k] = imgin_cv[int(i/2), int(j/2), k]
                    Statistics[0, 0] += 1
                    continue
                
                if ((i%2) == 0):
                    if ((j%2) == 0):
                        #print("1")
                        tmpedge = imgedge_cv_canny_L[int(i/2)-1:int(i/2)+1, int(j/2)-1:int(j/2)+1, k]
                        tmpin = imgin_cv[int(i/2)-1:int(i/2)+1, int(j/2)-1:int(j/2)+1, k].copy().astype(np.float64)
                        #print("tmpedge=", tmpedge)
                        #print("tmpin=", tmpin)
                        
                        # Non Edge Pixel Neighbor
                        if (np.sum(tmpedge) == 0):
                            imgout[i, j, k] = np.uint8(9/16*tmpin[1, 1] + 3/16*tmpin[0, 1] + 3/16*tmpin[1, 0] + 1/16*tmpin[0, 0])
                        # 1 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 255):
                            imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 2 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 510):
                            if ((tmpedge[1, 1] == 255) and (tmpedge[0, 0]== 255)):
                                imgout[i, j, k] = np.uint8(tmpin[1, 1])
                            elif ((tmpedge[0, 1] == 255) and (tmpedge[1, 0] == 255)):
                                imgout[i, j, k] = np.uint8(tmpin[0, 0])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 3 Edge Pixel Neighbor
                        else:
                            if (tmpedge[1, 1] == 0):
                                imgout[i, j, k] = np.uint8(tmpin[1, 1])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 255]))
                        
                        Statistics[1, 0] += 1
                    if ((j%2) == 1):
                        #print(2)
                        tmpedge = imgedge_cv_canny_L[int(i/2)-1:int(i/2)+1, int(j/2):int(j/2)+2, k]
                        tmpin = imgin_cv[int(i/2)-1:int(i/2)+1, int(j/2):int(j/2)+2, k].copy().astype(np.float64)
                        #print("tmpedge=", tmpedge)
                        #print("tmpin=", tmpin)
                        
                        # Non Edge Pixel Neighbor
                        if (np.sum(tmpedge) == 0):
                            imgout[i, j, k] = np.uint8(9/16*tmpin[1, 0] + 3/16*tmpin[0, 0] + 3/16*tmpin[1, 1] + 1/16*tmpin[0, 1])
                        # 1 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 255):
                            imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 2 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 510):
                            if ((tmpedge[1, 1] == 255) and (tmpedge[0, 0]== 255)):
                                imgout[i, j, k] = np.uint8(tmpin[1, 0])
                            elif ((tmpedge[0, 1] == 255) and (tmpedge[1, 0] == 255)):
                                imgout[i, j, k] = np.uint8(tmpin[0, 1])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 3 Edge Pixel Neighbor
                        else:
                            if (tmpedge[1, 1] == 0):
                                imgout[i, j, k] = np.uint8(tmpin[1, 0])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 255]))
                        
                        Statistics[2, 0] += 1
                if ((i%2) == 1):
                    if ((j%2) == 0):
                        #print(3)
                        tmpedge = imgedge_cv_canny_L[int(i/2):int(i/2)+2, int(j/2)-1:int(j/2)+1, k]
                        tmpin = imgin_cv[int(i/2):int(i/2)+2, int(j/2)-1:int(j/2)+1, k].copy().astype(np.float64)
                        #print("tmpedge=", tmpedge)
                        #print("tmpin=", tmpin)
                        
                        # Non Edge Pixel Neighbor
                        if (np.sum(tmpedge) == 0):
                            imgout[i, j, k] = np.uint8(9/16*tmpin[0, 1] + 3/16*tmpin[0, 0] + 3/16*tmpin[1, 1] + 1/16*tmpin[1, 0])
                        # 1 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 255):
                            imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 2 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 510):
                            if ((tmpedge[1, 1] == 255) and (tmpedge[0, 0]== 255)):
                                imgout[i, j, k] = np.uint8(tmpin[0, 1])
                            elif ((tmpedge[0, 1] == 255) and (tmpedge[1, 0] == 255)):
                                imgout[i, j, k] = np.uint8(tmpin[1, 0])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 3 Edge Pixel Neighbor
                        else:
                            if (tmpedge[1, 1] == 0):
                                imgout[i, j, k] = np.uint8(tmpin[0, 1])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 255]))
                        
                        Statistics[3, 0] += 1
                    if ((j%2) == 1):
                        #print(4)
                        tmpedge = imgedge_cv_canny_L[int(i/2):int(i/2)+2, int(j/2):int(j/2)+2, k]
                        tmpin = imgin_cv[int(i/2):int(i/2)+2, int(j/2):int(j/2)+2, k].copy().astype(np.float64)
                        #print("tmpedge=", tmpedge)
                        #print("tmpin=", tmpin)
                        
                        # Non Edge Pixel Neighbor
                        if (np.sum(tmpedge) == 0):
                            imgout[i, j, k] = np.uint8(9/16*tmpin[0, 0] + 3/16*tmpin[0, 1] + 3/16*tmpin[1, 0] + 1/16*tmpin[1, 1])
                        # 1 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 255):
                            imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 2 Edge Pixel Neighbor
                        elif (np.sum(tmpedge) == 510):
                            if ((tmpedge[1, 1] == 255) and (tmpedge[0, 0]== 255)):
                                imgout[i, j, k] = np.uint8(tmpin[1, 1])
                            elif ((tmpedge[0, 1] == 255) and (tmpedge[1, 0] == 255)):
                                imgout[i, j, k] = np.uint8(tmpin[0, 0])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 0]))
                        # 3 Edge Pixel Neighbor
                        else:
                            if (tmpedge[1, 1] == 0):
                                imgout[i, j, k] = np.uint8(tmpin[0, 0])
                            else:
                                imgout[i, j, k] = np.uint8(np.mean(tmpin[tmpedge == 255]))
                        
                        Statistics[4, 0] += 1
    clock.toc()
    
    imgout = imgout[:, :, [2, 1, 0]]
    
    img.imsave(r"D:\Final.bmp", imgout)
    
    # HES Only For 2x
    #imgout = imgout_cv.copy()
    #print("HES: ", end="")
    #clock.tic()
    #for i in np.arange(1, nvo-1, 1):
    #    for j in np.arange(1, nho-1, 1):
    #        # Edge Pixel
    #        if (imgedge_cv_canny_H[i, j]):
    #            imgout[i, j, :] = imgin_cv[int(i/2), int(j/2), :]
    #clock.toc()
    #
    #img.imsave(r"D:\Final.bmp", imgout)
    