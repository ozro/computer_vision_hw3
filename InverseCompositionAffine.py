import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1):
    # Input: 
    #	It: template image
    #	It1: Current image

    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = M0.copy()

    th = 0.15
    dp = np.ones(M.size)

    xx, yy = getMeshGrid(It.shape)
    Iy, Ix = np.gradient(It)
    Ix = Ix.flatten()
    Iy = Iy.flatten()
    mask = np.ones(It.shape)

    A = np.zeros((It.size, 6))
    A[:,0] = Ix * xx 
    A[:,1] = Ix * yy 
    A[:,2] = Ix
    A[:,3] = Iy * xx 
    A[:,4] = Iy * yy 
    A[:,5] = Iy 

    while np.linalg.norm(dp)>=th:
        # warp template, gradients by M
        warpedIt = cv2.warpAffine(It, M, (It.shape[1], It.shape[0]))
        warpedMask = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))
        overlap = warpedMask * It1

        # construct A and b
        b = (overlap - warpedIt).flatten()

        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        dM = np.zeros((3,3))
        dM[0:2, :] = M0 + dp.reshape(M0.shape)
        dM[2, :] = np.array([0.0,0.0,1.0])
        M = np.dot(M, np.linalg.inv(dM))

    return M

def getMeshGrid(shape):
    h, w = shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x,y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    return xx.flatten(),yy.flatten()
