import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    padding = int((size[0]-1)/2)
    res = np.empty((input_image.shape[0] + padding * 2, input_image.shape[1] + padding * 2, input_image.shape[2]))
    for i in range(input_image.shape[2]):
        res[:, :, i] = pad_reflect(input_image[:, :, i], padding)

    return res

def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    kernel_size = Kernel.shape[0]
    padded_img = reflect_padding(input_image, Kernel.shape)

    res = np.empty((padded_img.shape[0] - kernel_size + 1, padded_img.shape[1] - kernel_size + 1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1] - kernel_size):
            for x in range(padded_img.shape[0] - kernel_size):
                result = (padded_img[x:x+kernel_size, y:y+kernel_size, c] * Kernel).sum()
                res[x, y, c] = result

    return res

def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")

    kernel_size = size[0]
    padded_img = reflect_padding(input_image, size)

    res = np.empty((padded_img.shape[0] - kernel_size + 1, padded_img.shape[1] - kernel_size + 1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1] - kernel_size):
            for x in range(padded_img.shape[0] - kernel_size):
                result = np.median(padded_img[x:x + kernel_size, y:y + kernel_size, c])
                res[x, y, c] = result

    return res

def pad_reflect(input_image, padding):  # 2d pad
    res = np.empty((input_image.shape[0] + 2 * padding, input_image.shape[1] + 2 * padding))
    res_row, res_col = res.shape

    for row in range(padding):
        res[padding-row-1, padding:res_col-padding] = input_image[row+1, :]
    for row in range(input_image.shape[0]-padding-1, input_image.shape[0]-1):
        res[res_row-row+1, padding:res_col-padding] = input_image[row, :]

    for col in range(padding):
        res[padding:res_row-padding, padding-col-1] = input_image[:, col+1]
    for col in range(input_image.shape[1]-padding-1, input_image.shape[1]-1):
        res[padding:res_row-padding, res_col-col+1] = input_image[:, col]

    for row in range(padding):
        for col in range(padding):
            res[row, col] = input_image[padding-row, padding-col]
        for col in range(res_col-padding, res_col):
            res[row, col] = input_image[padding-row, res_col-col+1]

    for row in range(res_row-padding, res_row):
        for col in range(padding):
            res[row, col] = input_image[res_row-row+1, padding-col]
        for col in range(res_col-padding, res_col):
            res[row, col] = input_image[res_row-row+1, res_col-col+1]

    res[padding:input_image.shape[0] + padding, padding:input_image.shape[1] + padding] = input_image

    return res

def gkern(size, sigmax, sigmay):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussx = np.exp(-0.5 * np.square(ax) / np.square(sigmax))
    gaussy = np.exp(-0.5 * np.square(ax) / np.square(sigmay))
    kernel = np.outer(gaussx, gaussy)
    return kernel / np.sum(kernel)

def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    kernel_size = size[0]
    padded_img = reflect_padding(input_image, size)

    res = np.empty((padded_img.shape[0] - kernel_size + 1, padded_img.shape[1] - kernel_size + 1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1] - kernel_size):
            for x in range(padded_img.shape[0] - kernel_size):
                result = (padded_img[x:x+kernel_size, y:y+kernel_size, c] * gkern(kernel_size, sigmax, sigmay)).sum()
                res[x, y, c] = result

    return res


if __name__ == '__main__':
    # image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    # image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()


