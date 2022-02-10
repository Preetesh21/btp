import cv2
from matplotlib import pyplot as plt

def channels(images):
    fig, axs = plt.subplots(3, 3, constrained_layout=True,figsize=(10, 6))
    fig.suptitle('This is a somewhat long figure title', fontsize=16)
    for i in range(0,3):
        img=cv2.imread(images[i])
        b = img.copy()
        # set green and red channels to 0
        b[:, :, 1] = 0
        b[:, :, 2] = 0

        g = img.copy()
        # set blue and red channels to 0
        g[:, :, 0] = 0
        g[:, :, 2] = 0

        r = img.copy()
        # set blue and green channels to 0
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        
        axs[0,i].imshow(b)
        title="Image"+str(i+1) + "with G=0 and R=0"
        axs[0,i].set_title(title)
        axs[0,i].axis('off')

        axs[1,i].imshow(g)
        title="Image"+str(i+1) + "with B=0 and R=0"
        axs[1,i].set_title(title)
        axs[1,i].axis('off')

        axs[2,i].imshow(r)
        title="Image"+str(i+1) + "with G=0 and B=0"
        axs[2,i].set_title(title)
        axs[2,i].axis('off')
    plt.show()
