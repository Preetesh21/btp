from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from tensorflow.python.framework import ops
import keras.backend as K
from PIL import Image
import tensorflow as tf
import numpy as np
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython.display as display
from IPython.display import Image
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tf.compat.v1.disable_eager_execution()

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # avg
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # create heat map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def display_images():
    images = []
    for img_path in glob.glob('./img_grad_cam/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

def main():
    pic_folder = "./img/"
    pic_cam_folder = "./img_grad_cam/"
    model_vgg = VGG16(weights='imagenet')
    print(model_vgg.summary())
    list_name = os.listdir(pic_folder)

    arr_images = []
    for i, file_name in enumerate(list_name):
        img = load_image(pic_folder + file_name)
        predictions = model_vgg.predict(img)
        top_1 = decode_predictions(predictions)[0][0]
        print('Predicted class:')
        print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

        predicted_class = np.argmax(predictions)
        cam_image, heat_map = grad_cam(model_vgg, img, predicted_class, "block5_pool")

        img_file = image.load_img(pic_folder + list_name[i])
        img_file = image.img_to_array(img_file)

        # save img
        cam_image = cv2.resize(cam_image, (img_file.shape[1], img_file.shape[0]), cv2.INTER_LINEAR)
        cv2.putText(cam_image,str(top_1[1]), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))
        cv2.putText(cam_image,str(top_1[2]), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))

        cam_image = cam_image.astype('float32')
        im_h = cv2.hconcat([img_file, cam_image])
        cv2.imwrite(pic_cam_folder + list_name[i], im_h)

    display_images()
