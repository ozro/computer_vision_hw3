import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
    # Input: 
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	bases: [n, m, k] where nxm is the size of the template.
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    th = 0.001
    B = bases.reshape((bases.shape[0] * bases.shape[1], bases.shape[2]))
    Bn = (np.eye(B.shape[0]) - np.dot(B, B.T))

    p = np.zeros(2)
    dp = th
    spline_It = getSpline(It)
    spline_It1 = getSpline(It1)
    xx, yy = getMeshGrid(rect)

    #Generate template patch
    I = spline_It.ev(yy,xx).flatten()

    while np.linalg.norm(dp)>=th:
        #Generate patch and gradients
        I1 = spline_It1.ev(yy+p[1], xx+p[0]).flatten()
        I1x = spline_It1.ev(yy+p[1], xx+p[0], dy=1).flatten()
        I1y = spline_It1.ev(yy+p[1], xx+p[0], dx=1).flatten()

        #Calculate dp
        A = np.zeros((I1x.size,2))
        A[:, 0] = I1x
        A[:, 1] = I1y
        b = I - I1
        b = np.expand_dims(b, axis=1)
        A = np.dot(Bn, A)
        b = np.dot(Bn, b)
        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        p += dp.reshape((2,))

    return p
    
def getMeshGrid(rect):
    x1, y1, x2, y2 = rect
    x = np.arange(x1, x2+0.5)
    y = np.arange(y1, y2+0.5)
    xx, yy = np.meshgrid(x,y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    return xx,yy

def getSpline(im):
    h, w = im.shape
    x = np.arange(h)
    y = np.arange(w)
    return RectBivariateSpline(x,y,im)
