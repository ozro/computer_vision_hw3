import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

if __name__ == '__main__':
    data = np.load('../data/carseq.npy')
    frame = data[:, :, 0]
    rects = []
    rect = np.array([59, 116, 145, 151])
    rects.append(rect)
    for i in range(1, data.shape[2]):
        next_frame = data[:, :, i]
        p = LucasKanade(frame, next_frame, rect)

        rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rects.append(rect)


        if i in [1, 100, 200, 300, 400]:
            im = np.stack((next_frame,)*3, axis=-1)
            ax = plt.gca()
            ax.imshow(im)
            box = patches.Rectangle((rect[0],rect[3]),rect[2]-rect[0],rect[1]-rect[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(box)
            plt.savefig('q1-3-{}.png'.format(i))
            plt.show()
            box.remove()

        frame = next_frame

    rects = np.array(rects)
    np.save('carseqrects.npy', rects)