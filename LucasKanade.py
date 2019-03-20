import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input: 
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]
    
    # Put your implementation here
    p = p0
    th = 0.001

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
        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        p += dp

    return p

def getMeshGrid(rect):
    x1, y1, x2, y2 = rect
    x = np.arange(x1, x2)
    y = np.arange(y1, y2)
    xx, yy = np.meshgrid(x,y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    return xx,yy

def getSpline(im):
    h, w = im.shape
    x = np.arange(h)
    y = np.arange(w)
    return RectBivariateSpline(x,y,im)
