import numpy as np
import cv2
from scipy.ndimage import binary_erosion

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
    # Input:
    #	Images at time t and t+1 
    # Output:
    #	mask: [nxm]
    # put your implementation here
    
    th = 0.1
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2)
    # M = InverseCompositionAffine(image1, image2)

    warpedIm1 = cv2.warpAffine(image1, M, (image2.shape[1], image2.shape[0]))

    diff = np.absolute(warpedIm1 - image2)
    mask = (diff > th) & (warpedIm1 != 0)
    mask = binary_erosion(mask)

    return mask
