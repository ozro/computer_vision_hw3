import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2

# write your script here, we recommend the above libraries for making your animation
from SubtractDominantMotion import SubtractDominantMotion
if __name__ == '__main__':
    data = np.load('../data/aerialseq.npy')
    frame = data[:, :, 0]

    fig = plt.figure()
    ims = []

    for i in range(1, data.shape[2]):
        next_frame = data[:, :, i]
        mask = SubtractDominantMotion(frame, next_frame)

        img = np.stack((next_frame,)*3, axis=-1)
        img[:,:,1][mask] = 1 
        
        im = plt.imshow(img)
        ims.append([im])

        if i in [30, 60, 90, 120]:
            plt.savefig('q3-3-{}.png'.format(i))

        frame = next_frame

    im_ani = animation.ArtistAnimation(fig, ims, interval=30, repeat_delay=0, repeat=True, blit=True)
    plt.show()
