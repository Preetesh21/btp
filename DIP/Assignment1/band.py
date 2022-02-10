import imageio
from matplotlib import pyplot as plt

def band(images):
    fig, axs = plt.subplots(3, 3, constrained_layout=True,figsize=(10, 6))
    fig.suptitle('Image Bands', fontsize=16)
    for i in range(0,3):
        img=imageio.imread(images[i])
        axs[0,i].imshow(img[:,:,0])
        title="Image"+str(i+1)+"R Channel"
        axs[0,i].set_title(title)
        axs[0,i].axis('off')
    
    for i in range(0,3):
        img=imageio.imread(images[i])
        axs[1,i].imshow(img[:,:,1])
        title="Image"+str(i+1)+"B Channel"
        axs[1,i].set_title(title)
        axs[1,i].axis('off')
    
    for i in range(0,3):
        img=imageio.imread(images[i])
        axs[2,i].imshow(img[:,:,2])
        title="Image"+str(i+1)+"G Channel"
        axs[2,i].set_title(title)
        axs[2,i].axis('off')
    plt.show()