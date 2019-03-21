import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis

# write your script here, we recommend the above libraries for making your animation

if __name__ == '__main__':
    fig = plt.figure()
    ims = []
    data = np.load('../data/sylvseq.npy')
    bases = np.load('../data/sylvbases.npy')
    frame = data[:, :, 0]
    rects = []
    rect = np.array([101, 61, 155, 107])
    rects.append(rect)
    for i in range(1, data.shape[2]):
        next_frame = data[:, :, i]
        p = LucasKanadeBasis(frame, next_frame, rect, bases)

        rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rects.append(rect)

        im = np.stack((next_frame,)*3, axis=-1)
        ax = plt.gca()
        img = plt.imshow(im)
        box = patches.Rectangle((rect[0],rect[3]),rect[2]-rect[0],rect[1]-rect[3],linewidth=1,edgecolor='r',facecolor='none')
        patch = ax.add_patch(box)
        # if i in [1, 200, 300, 350, 400]:
            # plt.savefig('q2-3-{}.png'.format(i))
        ims.append([img, patch])

        frame = next_frame

    im_ani = animation.ArtistAnimation(fig, ims, interval=30, repeat_delay=0, repeat=True, blit=True)
    plt.show()
    rects = np.array(rects)
    np.save('sylvseqrects.npy', rects)