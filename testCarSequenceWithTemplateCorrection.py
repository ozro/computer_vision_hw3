import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

if __name__ == '__main__':
    fig = plt.figure()
    ims = []
    data = np.load('../data/carseq.npy')
    epsilon = 0.5
    frame = data[:, :, 0]
    frame0 = frame.copy()
    rects = []
    rect = np.array([59, 116, 145, 151])
    rect0 = rect.copy() 
    rects.append(rect)
    pt = (0,0)
    rects_n = np.load('carseqrects.npy')
    p_prev = None
    for i in range(1, data.shape[2]):
        next_frame = data[:, :, i]

        p = LucasKanade(frame, next_frame, rect)
        ps = LucasKanade(frame0, next_frame, rect0) #calculate with original template

        rect_next = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rects.append(rect_next)

        rects.append(rect)
        if(p_prev is None or np.linalg.norm(p-p_prev) < epsilon):
            frame = next_frame
            rect = rect_next
            p_prev = p.copy()

        im = np.stack((next_frame,)*3, axis=-1)
        ax = plt.gca()
        img = plt.imshow(im)
        box = patches.Rectangle((rect_next[0],rect_next[3]),rect_next[2]-rect_next[0],rect_next[1]-rect_next[3],linewidth=1,edgecolor='r',facecolor='none')
        box_n = patches.Rectangle((rects_n[i][0],rects_n[i][3]),rects_n[i][2]-rects_n[i][0],rects_n[i][1]-rects_n[i][3],linewidth=1,edgecolor='b',facecolor='none')
        patch1 = ax.add_patch(box)
        patch2 = ax.add_patch(box_n)
        if i in [1, 100, 200, 300, 400]:
            plt.savefig('q1-4-{}.png'.format(i))
        ims.append([img, patch1, patch2])
        
    im_ani = animation.ArtistAnimation(fig, ims, interval=30, repeat_delay=0, repeat=True, blit=True)
    plt.show()

    rects = np.array(rects)
    np.save('carseqrects-wcrt.npy', rects)