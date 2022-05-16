from pytictoc import TicToc
import numpy as np
import matplotlib.image as img
import cv2

clock = TicToc()

clock.tic()

Ratio = 2

GradientThreshold = 16

imgin = img.imread(r"..\Image\ImageOfTesting003.bmp")
(nvi, nhi, nci) = imgin.shape

imgout_NN = cv2.resize(imgin, (int(nhi*Ratio), int(nvi*Ratio)), interpolation=cv2.INTER_NEAREST)
img.imsave(r"ImgOut_NN.bmp", imgout_NN)

imgout_BiLinear = cv2.resize(imgin, (int(nhi*Ratio), int(nvi*Ratio)), interpolation=cv2.INTER_LINEAR)
img.imsave(r"ImgOut_BiLinear.bmp", imgout_BiLinear)

imgout = np.zeros((int(nvi*Ratio), int(nhi*Ratio), nci), dtype=np.uint8)
(nvo, nho, nco) = imgout.shape

for i in np.arange(nvo):
    for j in np.arange(nho):
        for k in np.arange(nco):
            y = (1/Ratio) * (i + 0.5)   # Y Cordinate Of ImgOut
            x = (1/Ratio) * (j + 0.5)   # X Cordinate of ImgOut
            ii = int(np.floor(y - 0.5))   # Y Index Of Left-Top Corner Of 2x2 Surrounding Pixel In ImgIn
            jj = int(np.floor(x - 0.5))   # X Index Of Left-Top Corner Of 2x2 Surrounding Pixel In ImgIn
            if ((ii < 0) and (jj < 0)):                 # Left-Top Corner Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii < 0) and (jj >= 0)):              # Top Row Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii < 0) and (jj >= (nhi-1))):          # Right-Top Corner Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii >= 0) and (jj < 0)):              # Left Column Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii >= 0) and (jj >= (nhi-1))):        # Right Column Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii >= (nvi-1)) and (jj < 0)):         # Left Bottom Corner Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii >= (nvi-1)) and (jj >= 0)):        # Bottom Raw Of Image
                #imgtmp = np.array([[], []])
                continue
            elif ((ii >= (nvi-1)) and (jj >= (nhi-1))):    # Right Bottom Corner Of Image
                continue
            else:                                       # Inner Area Of Image
                #print("(i, j, y, x, ii, jj) = ({}, {}, {}, {}, {}, {})".format(i, j, y, x, ii, jj))
                #print(imgin[ii:(ii+2), jj:(jj+2), k])
                ((S1, S2), (S3, S4)) = imgin[ii:(ii+2), jj:(jj+2), k].copy().astype(np.float64)
                D1 = np.abs(S1 - S2)
                D2 = np.abs(S1 - S3)
                D3 = np.abs(S1 - S4)
                D4 = np.abs(S2 - S3)
                D5 = np.abs(S2 - S4)
                D6 = np.abs(S3 - S4)
                p = x - jj - 0.5
                q = y - ii - 0.5
                #if (j == 1):
                #    print("(p, q) = ({}, {})".format(p, q))
                #    exit()
                if (        # Vertical Edge
                    ((np.min([D1, D2, D3, D4, D5, D6]) == D2) or (np.min([D1, D2, D3, D4, D5, D6]) == D5))\
                    and\
                    (D1 > GradientThreshold)\
                    and\
                    (D5 > GradientThreshold)\
                ):
                    if (p < 0.5):
                        imgout[i, j, k] = (1-q)*S1 + q*S3
                    else:
                        imgout[i, j, k] = (1-q)*S2 + q*S4
                elif (      # Horizontal Edge
                    ((np.min([D1, D2, D3, D4, D5, D6]) == D1) or (np.min([D1, D2, D3, D4, D5, D6]) == D6))\
                    and\
                    (D3 > GradientThreshold)\
                    and\
                    (D4 > GradientThreshold)\
                ):
                    if (q < 0.5):
                        imgout[i, j, k] = (1-p)*S1 + p*S2
                    else:
                        imgout[i, j, k] = (1-p)*S3 + p*S4
                elif (      # SW-NE Edge
                    (np.min([D1, D2, D3, D4, D5, D6]) == D3)\
                    and\
                    ((D1 > GradientThreshold) or (D6 > GradientThreshold))\
                ):
                    if ((0.5*p + 0.5*q - 0.25) < 0):      # Near Left Top Corner
                        imgout[i, j, k] = S1
                    elif ((1.5*p + 1.5*q - 2.25) > 0):    # Near Right Bottom Corner
                        imgout[i, j, k] = S4
                    else:       # On Edge
                        imgout[i, j, k] = 0.5*(p*S2 + (1-p)*S3) + 0.5*((1-q)*S2 + q*S3)
                elif (      # NW-SE Edge
                    (np.min([D1, D2, D3, D4, D5, D6]) == D4)\
                    and\
                    ((D1 > GradientThreshold) or (D6 > GradientThreshold))\
                ):
                    if ((-0.5*p + 0.5*q + 0.25) < 0):      # Near Right Top Corner
                        imgout[i, j, k] = S2
                    elif ((0.5*p - 0.5*q + 0.25) < 0):    # Near Left Bottom Corner
                        imgout[i, j, k] = S3
                    else:       # On Edge
                        imgout[i, j, k] = 0.5*(p*S4 + (1-p)*S1) + 0.5*((1-q)*S1 + q*S4)
                else:
                    imgout[i, j, k] = (1-p)*(1-q)*S1 + S2*p*(1-q) + S3*(1-p)*q + S4*p*q
                
                #imgout[i, j, k] = (1-p)*(1-q)*S1 + S2*p*(1-q) + S3*(1-p)*q + S4*p*q

img.imsave(r"ImgOut_2x2.bmp", imgout)

clock.toc()
