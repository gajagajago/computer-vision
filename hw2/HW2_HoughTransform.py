
import math
import glob
import numpy as np
from PIL import Image

#temp import
from scipy import signal

# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=2
highThreshold=0.03
lowThreshold=0.01
rhoRes=2
thetaRes=math.pi/180
nLines=20

def replicatePadding(Igs, G):
    # 1. Give padding to Igs, pad with nearest value
    padding_v = int((G.shape[0] - 1) / 2)
    padding_h = int((G.shape[1] - 1) / 2)

    padded_h = Igs.shape[0] + 2 * padding_v
    padded_w = Igs.shape[1] + 2 * padding_h
    padded_h_last_idx = padded_h - 1
    padded_w_last_idx = padded_w - 1
    iconv = np.zeros((padded_h, padded_w))

    # org array copy
    iconv[padding_v:padded_h - padding_v, padding_h:padded_w - padding_h] = Igs

    # pad with nearest value
    for row in range(padding_v):
        for col in range(padded_w):
            if col < padding_v:
                iconv[row][col] = iconv[padding_v][padding_h]
            elif padding_v <= col <= padded_w - padding_v:
                iconv[row][col] = iconv[padding_v][col]
            else:
                iconv[row][col] = iconv[padding_v][padded_w_last_idx - padding_h]

    for row in range(padding_v, padded_h - padding_v):
        for col in range(padding_h):
            iconv[row][col] = iconv[row][padding_h]
        for col in range(padded_w - padding_h, padded_w):
            iconv[row][col] = iconv[row][padded_w_last_idx - padding_h]

    for row in range(padded_h - padding_v, padded_h):
        for col in range(padded_w):
            if col < padding_v:
                iconv[row][col] = iconv[padded_h_last_idx - padding_v][padding_h]
            elif padding_v <= col <= padded_w - padding_v:
                iconv[row][col] = iconv[padded_h_last_idx - padding_v][col]
            else:
                iconv[row][col] = iconv[padded_h_last_idx - padding_v][padded_w_last_idx - padding_h]

    return iconv

def convolve(Img, G):
    oh = Img.shape[0] - G.shape[0] + 1
    ow = Img.shape[1] - G.shape[1] + 1
    out = np.zeros((oh, ow))

    for y in range(Img.shape[1]-G.shape[1]+1):
        for x in range(Img.shape[0]-G.shape[0]+1):
            result = (Img[x:x+G.shape[0], y:y+G.shape[1]] * np.flip(G)).sum()
            out[x][y] = result

    return out

def ConvFilter(Igs, G):
    iconv = replicatePadding(Igs, G)
    out = convolve(iconv, G)

    return out
    # return out.astype(np.uint8).clip(0, 255)

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # 1. Smooth Igs w/ Gaussian kernel w/ sigma

    # set arbitrary g_kernel for test
    g_kernel = gkern((5, 5), sigma, sigma)

    # Replicate pad Igs and Convolve w/ g_kernel
    smoothed = ConvFilter(Igs, g_kernel).clip(0, 1)
    print("Smoothed: ", smoothed, "\n")
    print("Max: ", np.amax(smoothed), " Min: ", np.amin(smoothed), "\n")

    # 2. Convolve smoothed img w/ sobel operators => Ix, Iy
    sobel_g = np.array([[1], [2], [1]])
    sobel_d = np.array([[-1, 0, 1]])

    Ix = convolve(convolve(smoothed, sobel_g), sobel_d).clip(0, 1)
    Iy = convolve(convolve(smoothed, sobel_g.T), sobel_d.T).clip(0, 1)

    print("Ix: ", Ix, " Iy: ", Iy, "\n");
    # 3. Get gradient magnitude img, gradient direction img Io
    # Imag = np.sqrt(np.sum(np.square(Ix), np.square(Iy))).asType(np.uint8).clip(0, 255)
    Imag = np.sqrt(np.add(np.square(Ix), np.square(Iy))).clip(0, 1)
    print("Max: ", np.amax(Imag), " Min: ", np.amin(Imag), "\n")

    Io = np.arctan2(Iy, Ix)

    # 4. Apply non maximal suppression
    # 5. Apply double thresholding


    # 잠시 주석처리
    # return Im, Io, Ix, Iy

    return Imag, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...


    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...


    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...


    return l

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

    rows = np.zeros((size[0], 1), dtype=np.float16)
    cols = np.zeros((1, size[1]), dtype=np.float16)

    for i in range(size[0]):
        rows[i][0] = gauss_seperable((i-(size[0]-1)/2), sigmax)
    for j in range(size[1]):
        cols[0][j] = gauss_seperable((j-(size[1]-1)/2), sigmay)

    return (rows*cols) / np.sum(rows*cols)

def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # 임시 print
        print("Igs: ", Igs, "\n")
        print("Max: ", np.amax(Igs), " Min: ", np.amin(Igs), "\n")

        # Hough function

        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)

        # 밑에 임시 주석처리

        # H= HoughTransform(Im, rhoRes, thetaRes)
        # lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        # l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()