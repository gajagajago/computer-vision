import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import math

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    # Form affine matrix M
    p1, p2, p3, p4, p5, p6 = p
    M = np.array([[1+p1, p3, p5], [p2, 1+p4, p6]])  # (2,3) matrix

    # img1: I(t)
    img1_h, img1_w = img1.shape

    # See how template I(t) --M--> I(t+1) would have transformed
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

    # Error image T(x) - I(W(x;p))
    Err = img1 - I_W

    # Parametric model
    Hessian = np.zeros((6, 6))
    ParamSum = np.zeros((6, 1))

    for x in range(Err.shape[0]):
        for y in range(Err.shape[1]):
            D_I = np.array([[Gx[x][y], Gy[x][y]]]) # (1,2)
            Jac = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]]) # (2,6)
            err = Err[x][y]
            ret = np.dot(np.dot(D_I, Jac).T, err)
            H = np.dot(np.dot(D_I, Jac).T, np.dot(D_I, Jac))

            ParamSum += ret
            Hessian += H

    Hessian_inv = np.linalg.inv(Hessian)
    dp = np.dot(Hessian_inv, ParamSum).reshape(6)

    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this

    p = np.zeros(6)

    # M = np.array([[p1, p3, p5], [p2, p4, p6]]) # (2,3) matrix
    dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
    p += dp
    p1, p2, p3, p4, p5, p6 = p
    M = np.array([[1+p1, p3, p5], [p2, 1+p4, p6]])  # (2,3) matrix

    # img1: I(t)
    img1_h, img1_w = img1.shape

    # Warping img1
    img1_origin = np.array([0, 0, 1])
    img1_end = np.array([img1_w-1, img1_h-1, 1])

    warped_origin_x, warped_origin_y = np.dot(M, img1_origin)
    warped_end_x, warped_end_y = np.dot(M, img1_end)

    # Create meshgrid of the warped I(t)
    warped_x_ls = np.linspace(warped_origin_x, warped_end_x, img1_w)
    warped_y_ls = np.linspace(warped_origin_y, warped_end_y, img1_h)
    warped_x_mg, warped_y_mg = np.meshgrid(warped_x_ls, warped_y_ls)

    # RectBivariateSpline on img1
    img1_x_ls = np.linspace(0, img1_w-1, img1_w)
    img1_y_ls = np.linspace(0, img1_h-1, img1_h)
    rect_B_spline = RectBivariateSpline(img1_x_ls, img1_y_ls, img1.T)

    # I(W(x;p))
    warped_img1 = rect_B_spline.ev(warped_x_mg, warped_y_mg)

    # |I(t+1) - I(t)|
    moving_image = np.abs(img2 - warped_img1)

    # tuned parameters to neglect camera calibration
    th_hi = 0.3 * 256
    th_lo = 0.25 * 256

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)

    # plt.figure()
    # plt.axis('off')
    # plt.subplot(2, 2, 1)
    # plt.imshow(warped_img1.astype(np.uint8))
    # plt.xlabel('Warped Img1')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(img2.astype(np.uint8))
    # plt.xlabel('Img2')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(moving_image.astype(np.uint8))
    # plt.xlabel('Moving Img')
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(hyst.astype(np.uint8))
    # plt.xlabel('Thresholded Img')
    #
    # plt.show()

    return hyst

if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    