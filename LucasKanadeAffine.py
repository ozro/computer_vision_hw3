import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
    # Input: 
    #	It: template image
    #	It1: Current image
    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros((1,6))
    M = M0

    th = 0.015
    dp = np.ones(p.shape)

    xx, yy = getMeshGrid(It.shape)
    Iy, Ix = np.gradient(It)
    mask = np.ones(It.shape)

    A = np.zeros((It.size, 6))

    while np.linalg.norm(dp)>=th:
        # warp template, gradients by M
        warpedIt = cv2.warpAffine(It, M, (It.shape[1], It.shape[0]))
        warpedIx = cv2.warpAffine(Ix, M, (It.shape[1], It.shape[0])).flatten()
        warpedIy = cv2.warpAffine(Iy, M, (It.shape[1], It.shape[0])).flatten()
        warpedMask = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))
        overlap = warpedMask * It1

        # construct A and b
        A[:,0] = warpedIx * xx 
        A[:,1] = warpedIx * yy 
        A[:,2] = warpedIx
        A[:,3] = warpedIy * xx 
        A[:,4] = warpedIy * yy 
        A[:,5] = warpedIy 
        b = (warpedIt - overlap).flatten()

        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        p += dp

        M = M0 + p.reshape(M0.shape)
    return M

def getMeshGrid(shape):
    h, w = shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x,y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    return xx.flatten(),yy.flatten()