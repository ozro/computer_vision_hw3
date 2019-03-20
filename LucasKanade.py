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

    return p

def getPatchInterp(im, rect, p):
	h, w = im.shape
	x = np.arange(0,h)
	y = np.arange(0,w)
	spline = RectBivariateSpline(x,y,im)
	return spline
