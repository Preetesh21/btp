import cv2
import numpy as np
from matplotlib import pyplot as plt
from clip import clip_image


def operations(images):
    for i in range(0,3):
        fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(10, 6))
        fig.suptitle('Image Operations', fontsize=16)
        img=cv2.imread(images[i],0)

        # Transformation
        linear_image = linear_transformation(img.copy(), 0.5, 10)
        logarith_image = logarithm_transformation(img.copy(), 1.5, 10)
        exponential_image = exponential_transformation(img.copy(), 1, -1)
        gamma_image = gamma_transformation(img.copy(), 1, -1, 0.5)

        axs[0,0].imshow(linear_image)
        title="Linear Transformation on Image"+str(i+1)
        axs[0,0].set_title(title)
        axs[0,0].axis('off')

        axs[0,1].imshow(logarith_image)
        title="Logarithmic Transformation on Image"+str(i+1)
        axs[0,1].set_title(title)
        axs[0,1].axis('off')

        axs[1,0].imshow(exponential_image)
        title="Exponential Transformation on Image"+str(i+1)
        axs[1,0].set_title(title)
        axs[1,0].axis('off')

        axs[1,1].imshow(gamma_image)
        title="Gamma Transformation on Image"+str(i+1)
        axs[1,1].set_title(title)
        axs[1,1].axis('off')
        plt.show()
        # To ascertain total numbers of rows and 
        # columns of the image, size of the image
        m, n = img.shape

        plot_histogram(img)

        # Transformation to obtain stretching
        constant = (255-0)/(img.max()-img.min())
        img_stretch = img * constant
        plot_histogram(img_stretch)

        _,equ_img=histogram_equilization(img)
        fig, axs = plt.subplots(1, 2, constrained_layout=True,figsize=(10, 6))
        fig.suptitle('Histogram Operations', fontsize=16)
        # Storing stretched Image
        axs[0].imshow(img_stretch)
        title="Histogram Stretching on Image"+str(i+1)
        axs[0].set_title(title)
        axs[0].axis('off')

        axs[1].imshow(equ_img)
        title="Histogram Equilization on Image"+str(i+1)
        axs[1].set_title(title)
        axs[1].axis('off')
        plt.show()
        

def histogram_equilization(img):
    # creating a Histograms Equalization
    # of a image using cv2.equalizeHist()
    equ = cv2.equalizeHist(img)
    return (img,equ)

def plot_histogram(image):
    """Plot histogram of image
    Parameters
    ----------
    image : np.ndarray BGR h,w,c dimentional
        image
    """
    hist, _ = np.histogram(image, bins=256)
    plt.bar(np.arange(256), hist)
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    plt.title('Histogram Plot')
    plt.show()



# function to obtain histogram of an image
def hist_plot(img):
    # empty list to store the count 
    # of each intensity value
    count =[]
    # empty list to store intensity 
    # value
    r = []
    m, n = img.shape
    # loop to traverse each intensity 
    # value
    for k in range(0, 256):
        r.append(k)
        count1 = 0
          
        # loops to traverse each pixel in 
        # the image 
        for i in range(m):
            for j in range(n):
                if img[i, j]== k:
                    count1+= 1
        count.append(count1)
          
    return (r, count)


def linear_transformation(image, alpha, beta):
    """Linear transformation
    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    return clip_image((alpha * image + beta).astype(np.uint8))


def logarithm_transformation(image, alpha, beta):
    """Logarithm transformation
    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.log2(1 + image/255)) + beta) * 255
    return clip_image(image.astype(np.uint8))


def exponential_transformation(image, alpha, beta):
    """Exponential transformation
    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.exp(image/255)) + beta) * 255
    return clip_image(image.astype(np.uint8))


def gamma_transformation(image, gamma, alpha=255, beta=0):
    """Power-law transformation
    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    gamma: float
        gamma parameter
    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.power(image/255, gamma)) + beta) * 255
    return clip_image(image.astype(np.uint8))