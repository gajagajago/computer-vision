import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import math

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###

    # Form affine matrix M
    p1, p2, p3, p4, p5, p6 = p
    M = np.array([[1+p1, p3, p5], [p2, 1+p4, p6]])  # (2,3) matrix

    # img1: I(t)
    img1_h, img1_w = img1.shape

    # See how template I(t) --p--> I(t+1) would have transformed
    img1_origin = np.array([0, 0, 1])
    img1_end = np.array([img1_w-1, img1_h-1, 1])

    warped_origin_x, warped_origin_y = np.dot(M, img1_origin)
    warped_end_x, warped_end_y = np.dot(M, img1_end)

    # Create meshgrid of the warped I(t)
    warped_x_ls = np.linspace(warped_origin_x, warped_end_x, img1_w)
    warped_y_ls = np.linspace(warped_origin_y, warped_end_y, img1_h)
    warped_x_mg, warped_y_mg = np.meshgrid(warped_x_ls, warped_y_ls)

    # img2: I(t+1)
    img2_h, img2_w = img2.shape

    # RectBivariateSpline on img2
    img2_x_ls = np.linspace(0, img2_w-1, img2_w)
    img2_y_ls = np.linspace(0, img2_h-1, img2_h)
    rect_B_spline = RectBivariateSpline(img2_x_ls, img2_y_ls, img2.T)

    # I(W(x;p))
    I_W = rect_B_spline.ev(warped_x_mg, warped_y_mg)

    if I_W is not None:
        plt.figure()
        plt.imshow(I_W.astype(np.uint8))
        plt.axis()
        plt.show()

    # Error image T(x) - I(W(x;p))
    Err = img1 - I_W

    if Err is not None:
        plt.figure()
        plt.imshow(Err.astype(np.uint8))
        plt.axis()
        plt.show()



    dp = 0


    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this
    
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    moving_image = np.abs(img2 - img1) # you should delete this
    
    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this

    
    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":

    im1 = Image.open('data/organized-0.jpg').convert("L")
    im1_in = np.array(im1)

    im2 = Image.open('data/organized-1.jpg').convert("L")
    im2_in = np.array(im2)

    print(im1_in.shape)

    wim = lucas_kanade_affine(im1_in, im2_in, np.array([0.1, 0, 0, 0, 0.1, -0.2]), 0, 0)
    # data_dir = 'data'
    # video_path = 'motion.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    # tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    # T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    # for i in range(0, 50):
    #     img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
    #     I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    #     clone = I.copy()
    #     moving_img = subtract_dominant_motion(T, I)
    #     clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    #     clone[moving_img, 2] = 522
    #     out.write(clone)
    #     T = I
    # out.release()
    