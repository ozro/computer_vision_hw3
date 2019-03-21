import numpy as np
import cv2

from LucasKanadeAffine import LucasKanadeAffine

def SubtractDominantMotion(image1, image2):
    # Input:
    #	Images at time t and t+1 
    # Output:
    #	mask: [nxm]
    # put your implementation here
    
    th = 0.1
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2)

    warpedIm1 = cv2.warpAffine(image1, M, (image2.shape[1], image2.shape[0]))

    diff = np.absolute(warpedIm1 - image2)
    mask = (diff > th) & (warpedIm1 != 0)

    return mask
