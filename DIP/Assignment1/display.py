import imageio
from matplotlib import pyplot as plt
 
# Read RGB image

def display_image(images):
    fig, axs = plt.subplots(1, 3, constrained_layout=True,figsize=(10, 6))
    fig.suptitle('This is a somewhat long figure title', fontsize=16)
    for i in range(0,3):
        img=imageio.imread(images[i])
        print(img.shape)
        axs[i].imshow(img)
        title="subplot"+str(i+1)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.show()
    