import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """
    pyramid = []
    g = input_image.copy()
    pyramid.append(g)

    for lv in range(level):
        g = utils.down_sampling(g)
        pyramid.append(g)

    print("Gaussian\n")
    for lv in pyramid:
        print("shape: ", lv.shape, "\n")
    return pyramid


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """
    pyramid = []

    for lv in range(len(gaussian_pyramid)-1):
        dog = utils.safe_subtract(gaussian_pyramid[lv], utils.up_sampling(gaussian_pyramid[lv+1]))
        pyramid.append(dog)
    pyramid.append(gaussian_pyramid[-1])

    print("Laplacian\n")
    for lv in pyramid:
        print("shape: ", lv.shape, "\n")

    return pyramid

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    la = laplacian_pyramid(gaussian_pyramid(image1, level))
    lb = laplacian_pyramid(gaussian_pyramid(image2, level))

    gr = gaussian_pyramid(mask, level)

    ls = []
    for i in range(len(la)):
        ls.append((1-gr[i])*la[i] + gr[i]*lb[i])

    for lv in range(len(ls)-1, 0, -1):
        ls[lv-1] = utils.safe_add(ls[lv-1], utils.up_sampling(ls[lv]))

    return ls[0]

if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3

    # plt.figure()
    # plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    # plt.axis('off')
    # plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    # plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
