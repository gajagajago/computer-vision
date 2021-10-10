
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


def ConvFilter(Igs, G):
    # 1. Give padding to Igs, pad with nearest value
    padding_v = int((G.shape[0] - 1) / 2)
    padding_h = int((G.shape[1] - 1) / 2)

    padded_h = Igs.shape[0] + 2*padding_v
    padded_w = Igs.shape[1] + 2*padding_h
    padded_h_last_idx = padded_h-1
    padded_w_last_idx = padded_w-1
    iconv = np.zeros((padded_h, padded_w))

    out = np.zeros(Igs.shape)

    # org array copy
    iconv[padding_v:padded_h-padding_v, padding_h:padded_w-padding_h] = Igs

    # pad with nearest value
    for row in range(padding_v):
        for col in range(padded_w):
            if col < padding_v:
                iconv[row][col] = iconv[padding_v][padding_h]
            elif padding_v <= col <= padded_w-padding_v:
                iconv[row][col] = iconv[padding_v][col]
            else:
                iconv[row][col] = iconv[padding_v][padded_w_last_idx-padding_h]

    for row in range(padding_v, padded_h-padding_v):
        for col in range(padding_h):
            iconv[row][col] = iconv[row][padding_h]
        for col in range(padded_w-padding_h, padded_w):
            iconv[row][col] = iconv[row][padded_w_last_idx-padding_h]

    for row in range(padded_h-padding_v, padded_h):
        for col in range(padded_w):
            if col < padding_v:
                iconv[row][col] = iconv[padded_h_last_idx-padding_v][padding_h]
            elif padding_v <= col <= padded_w-padding_v:
                iconv[row][col] = iconv[padded_h_last_idx-padding_v][col]
            else:
                iconv[row][col] = iconv[padded_h_last_idx-padding_v][padded_w_last_idx-padding_h]

    # # temp
    # iconv_for_scipy = iconv.copy()

    # 2. Convolve with G
    for y in range(padded_w-G.shape[1]+1):
        for x in range(padded_h-G.shape[0]+1):
            result = (iconv[x:x+G.shape[0], y:y+G.shape[1]] * np.flip(G)).sum()
            out[x][y] = result

    # # Compare with scipy result
    # f = signal.convolve2d(iconv_for_scipy, G, 'valid')
    #
    # equal = np.array_equal(f, out)
    # print("Equal? ", equal, "\n")

    return out.clip(0, 255)

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...


    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...


    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...


    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...


    return l

def main():
    Igs = np.arange(100).reshape(10,-1)
    out = ConvFilter(Igs, np.arange(9).reshape(3, 3))

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        # Igs = np.array(img)
        # Igs = Igs / 255.

        out = ConvFilter(Igs, np.arange(9).reshape(3, 3))

        # Hough function
        # 밑에 임시 주석처리

        # Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        # H= HoughTransform(Im, rhoRes, thetaRes)
        # lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        # l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()