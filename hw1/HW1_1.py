import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of 2 int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    input_h, input_w, input_c = input_image.shape
    filter_h, filter_w = size
    padding_vert = int((filter_h-1)/2)
    padding_horiz = int((filter_w-1)/2)
    padding = (padding_vert, padding_horiz)

    res = np.empty((input_h + 2*padding_vert, input_w + 2*padding_horiz, input_c))

    for c in range(input_c):
        res[:, :, c] = pad_reflect(input_image[:, :, c], padding)

    return res

def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """
    padded_img = reflect_padding(input_image, Kernel.shape)

    res = np.empty((padded_img.shape[0]-Kernel.shape[0]+1, padded_img.shape[1]-Kernel.shape[1]+1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1] - Kernel.shape[1]):
            for x in range(padded_img.shape[0] - Kernel.shape[0]):
                result = ( padded_img[x:x+Kernel.shape[0], y:y+Kernel.shape[1], c] * np.flip(Kernel) ).sum()
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

    kernel_h, kernel_w = size
    padded_img = reflect_padding(input_image, size)

    res = np.empty((padded_img.shape[0]-kernel_h+1, padded_img.shape[1]-kernel_w+1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1] - kernel_w):
            for x in range(padded_img.shape[0] - kernel_h):
                result = np.median(padded_img[x:x+kernel_h, y:y+kernel_w, c])
                res[x, y, c] = result

    return res

def pad_reflect(input_2d, padding):
    """
    Args:
        input 2d array (numpy array)
        padding (tuple of 2 int [padding_vert, padding_horiz])
    Return:
        reflect padded 2d array
    """
    input_h, input_w = input_2d.shape
    padding_vert, padding_horiz = padding
    res = np.empty((input_h + 2*padding_vert, input_w + 2*padding_horiz))
    res_row, res_col = res.shape

    for row in range(1, 1+padding_vert):
        res[padding_vert-row, padding_horiz:res_col-padding_horiz] = input_2d[row, :] # upper row
        res[padding_vert-row, 0:padding_horiz] = np.flip(input_2d[row, 1:1+padding_horiz].reshape(1, -1), 1) # upper row left corner
        res[padding_vert-row, res_col-padding_horiz:res_col] = np.flip(input_2d[row, (input_w-1)-padding_horiz:input_w-1].reshape(1, -1), 1) # upper row right corner

        res[(padding_vert+input_h-1)+row, padding_horiz:res_col-padding_horiz] = input_2d[(input_h-1)-row, :] # lower row
        res[(padding_vert+input_h-1)+row, 0:padding_horiz] = np.flip(input_2d[(input_h-1)-row, 1:1 + padding_horiz].reshape(1, -1), 1)  # lower row left corner
        res[(padding_vert+input_h-1)+row, res_col - padding_horiz:res_col] = np.flip(input_2d[(input_h-1)-row, (input_w-1)-padding_horiz:input_w-1].reshape(1, -1), 1)  # lower row right corner

    for col in range(1, 1+padding_horiz):
        res[padding_vert:res_row-padding_vert, padding_horiz-col] = input_2d[:, col] # left col
        res[padding_vert:res_row-padding_vert, (padding_horiz+input_w-1)+col] = input_2d[:, (input_w-1)-col] # right col

    res[padding_vert:res_row-padding_vert, padding_horiz:res_col-padding_horiz] = input_2d # org array copy

    return res

def gkern(size, sigmax, sigmay):
    """
    Args:
        size: kernel size in tuple
        sigmax: x sigma
        sigmay: y sigma
    Returns:
        gaussian_kernel
    """
    def gauss_seperable(i, sigma):
        return np.exp(-0.5 * np.square(i) / np.square(sigma))

    rows = np.zeros((size[0], 1))
    cols = np.zeros((1, size[1]))

    for i in range(size[0]):
        rows[i][0] = gauss_seperable((i-(size[0]-1)/2), sigmax)
    for j in range(size[1]):
        cols[0][j] = gauss_seperable((j-(size[1]-1)/2), sigmay)

    return (rows*cols) / np.sum(rows*cols)

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
    padded_img = reflect_padding(input_image, size)
    kernel_h, kernel_w = size

    res = np.empty((padded_img.shape[0]-kernel_h+1, padded_img.shape[1]-kernel_w+1, padded_img.shape[2]))

    for c in range(padded_img.shape[2]):
        for y in range(padded_img.shape[1]-kernel_w):
            for x in range(padded_img.shape[0]-kernel_h):
                result = ( padded_img[x:x+kernel_h, y:y+kernel_w, c] * gkern(size, sigmax, sigmay) ).sum()
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
    sigmax, sigmay = 3, 3
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