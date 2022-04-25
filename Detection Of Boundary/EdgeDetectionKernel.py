# ================================================== #
#   Importation of Default Module
# ================================================== #

# ================================================== #
#   Importation of 3rd Party Module
# ================================================== #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

# ================================================== #
#   Importation of Self Development Module
# ================================================== #

# ================================================== #
#   Declaration AND Definition Of This Module Variable
# ================================================== #
fdir = r"./../Image"
fname = r"IMG_0043.PNG"
#fpath = fdir + "\\" + fname

# ================================================== #
#   Declaration AND Definition Of This Module Function
# ================================================== #

def ReadTestingImage (fdir: str=fdir, fname: str=fname) -> np.ndarray:
    fpath = fdir + "\\" + fname
    
    #imgin = img.imread(fpath)
    imgin = Image.open(fpath)
    
    return np.array(imgin)

# ================================================== #
#   Declaration AND Definition Of This Module Class
# ================================================== #

# ================================================== #
#   Testing Of This Module
# ================================================== #
if (__name__ == "__main__"):
    imgin = ReadTestingImage()
    print(imgin.shape)
    print(imgin.dtype)
    
    plt.imshow(imgin)
    plt.show()