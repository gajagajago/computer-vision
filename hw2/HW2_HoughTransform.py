
import math
import glob
import numpy as np
from PIL import Image

#temp import
from scipy import signal
from matplotlib import pyplot as plt

# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=3
highThreshold=0.04
lowThreshold=0.03
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

def isMax3(target, compare1, compare2):
    return target > compare1 and target > compare2

def isMaxN(target, compareList):
    ret = True

    for cmp in compareList:
        if target <= cmp:
            ret = False
            break

    return ret

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
    # print("Smoothed: ", smoothed, "\n")
    # print("Max: ", np.amax(smoothed), " Min: ", np.amin(smoothed), "\n")

    # 2. Convolve smoothed img w/ sobel operators => Ix, Iy
    sobel_g = np.array([[1], [2], [1]])
    sobel_d = np.array([[-1, 0, 1]])

    Ix = convolve(convolve(smoothed, sobel_g), sobel_d) # 이건 clip 하면 안됨
    Iy = convolve(convolve(smoothed, sobel_g.T), sobel_d.T) # 이건 clip 하면 안됨

    # print("Ix: ", Ix, " \nIy: ", Iy, "\n");
    # 3. Get gradient magnitude img, gradient direction img Io
    # Imag = np.sqrt(np.sum(np.square(Ix), np.square(Iy))).asType(np.uint8).clip(0, 255)
    Imag = np.sqrt(np.add(np.square(Ix), np.square(Iy))).clip(0, 1)
    # print("Max: ", np.amax(Imag), " Min: ", np.amin(Imag), "\n")

    Io = np.arctan2(Iy, Ix).astype(np.float16)
    # print("Io: ", Io, "\n")
    # print("Max: ", np.amax(Io), " Min: ", np.amin(Io), "\n")

    # 4. Apply non maximal suppression
    # !important Corner cases

    for x in range(Io.shape[0]):
        for y in range(Io.shape[1]):
            theta = Io[x][y]
            target_g = Imag[x][y]

            # 수평 방향
            if (-1/8)*math.pi <= theta <= (1/8)*math.pi or (theta >= (7/8)*math.pi) or (theta <= (-7/8)*math.pi) :
                l = Imag[x][y-1] if y>0 else 0
                r = Imag[x][y+1] if y<Io.shape[1]-1 else 0
                if not isMax3(target_g, l, r):
                    Imag[x][y] = 0

            # 우상향 대각 방향
            elif (1/8)*math.pi < theta < (3/8)*math.pi or (-7/8)*math.pi < theta < (-5/8)*math.pi :
                ld = Imag[x+1][y-1] if y>0 and x<Io.shape[0]-1 else 0
                ru = Imag[x-1][y+1] if y<Io.shape[1]-1 and x>0 else 0
                if not isMax3(target_g, ld, ru):
                    Imag[x][y] = 0

            # 수직 방향
            elif (3/8)*math.pi <= theta <= (5/8)*math.pi or (-5/8)*math.pi < theta < (-3/8)*math.pi:
                u = Imag[x-1][y] if x>0 else 0
                d = Imag[x+1][y] if x<Io.shape[0]-1 else 0
                if not isMax3(target_g, u, d):
                    Imag[x][y] = 0

            # 좌상향 대각 방향
            else :
                lu = Imag[x-1][y-1] if x>0 and y>0 else 0
                rd = Imag[x+1][y+1] if x<Io.shape[0]-1 and y<Io.shape[1]-1 else 0
                if not isMax3(target_g, lu, rd):
                    Imag[x][y] = 0
    # print(Io)
    # print("\n\n\n")
    # print(Imag)

    # print("Mean of Imag: ", np.mean(Imag), "\n")
    # 5. Apply double thresholding
    # !important Corner cases
    for x in range(Imag.shape[0]):
        for y in range(Imag.shape[1]):
            target = Imag[x][y]

            if target >= highThreshold:
                continue
            elif target < lowThreshold:
                Imag[x][y] = 0
            else:
                neighbors = Imag[max(x-1,0):min(Imag.shape[0], x+2), max(y-1,0):min(Imag.shape[1], y+2)]
                if np.max(neighbors) < highThreshold:
                    Imag[x][y] = 0

    # 잠시 주석처리
    # return Im, Io, Ix, Iy

    return Imag, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # 1. Fix origin to be Cx, Cy of Im
    # Assume Im.shape = (even, even)
    # ex. Im.shape = (10,10) -> Origin: Upper left intersection of 5th row and column
    Cx = Im.shape[0] / 2
    Cy = Im.shape[1] / 2

    # 2. Calculate rhoMax
    rhoMax = math.sqrt(Cx**2 + Cy**2)

    # 3. Make H = ( math.ceil(rhoMax/rhoRes), math.ceil(2*pi/thetaRes) )
    H = np.zeros((math.ceil(rhoMax/rhoRes), math.ceil(2*math.pi/thetaRes)))
    print("Shape of H: ", H.shape, "\n")

    # 4. For every non zero pixel in Im, calculate theta, rho -> Vote on H
    for y in range(Im.shape[1]):
        for x in range(Im.shape[0]):
            if Im[x][y] != 0:
                theta = np.arctan2(y-Cy, x-Cx)
                theta = theta if theta >= 0 else 2*math.pi + theta
                rho = (x-Cx)*np.cos(theta) + (y-Cy)*np.sin(theta)

                theta_normalized = math.floor(theta/thetaRes)
                rho_normalized = math.floor(rho/rhoRes)

                H[rho_normalized][theta_normalized] += 1
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # 1. Init returned list
    lRho = []
    lTheta = []

    # 2. Non maximal suppression
    for x in range(H.shape[0]):
        for y in range(H.shape[1]):
            lu = H[x-1][y-1] if x > 0 and y > 0 else 0
            u = H[x-1][y] if x > 0 else 0
            ru = H[x-1][y+1] if x > 0 and y < H.shape[1] - 1 else 0
            l = H[x][y-1] if y > 0 else 0
            r = H[x][y+1] if y < H.shape[1] - 1 else 0
            ld = H[x+1][y-1] if x < H.shape[0] - 1 and y > 0 else 0
            d = H[x+1][y] if x < H.shape[0] - 1 else 0
            rd = H[x+1][y+1] if x < H.shape[0] - 1 and y < H.shape[1] - 1 else 0

            isMax = isMaxN(H[x][y], [lu,u,ru,l,r,ld,d,rd])

            if not isMax:
                H[x][y] = 0

    # 3. Retrieve top n voted entries from H
    for i in range(nLines):
        normal_rho, normal_theta = np.unravel_index(H.argmax(), H.shape)
        lRho.append(normal_rho)
        lTheta.append(normal_theta)

        # print("CURR MAX in H: ", H.max(), "\n")
        # print("Received Max: ", H[normal_rho][normal_theta], "\n")

        H[normal_rho][normal_theta] = 0

    # 4. Restore original rho, theta value
    lRho = [rho * rhoRes for rho in lRho]
    lTheta = [theta * thetaRes for theta in lTheta]

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

        print(Igs.shape)
        # 임시 print
        # print("Igs: ", Igs, "\n")
        # print("Max: ", np.amax(Igs), " Min: ", np.amin(Igs), "\n")

        # Hough function

        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)

        # ret = (Im * 255).astype(np.uint8)
        # if ret is not None:
        #     plt.figure()
        #     plt.imshow(ret.astype(np.uint8))
        #     plt.axis('off')
        #     plt.show()


        H= HoughTransform(Im, rhoRes, thetaRes)
        # 주석

        # if H is not None:
        #     plt.figure()
        #     plt.imshow(H)
        #     plt.axis('off')
        #     plt.show()


        lRho,lTheta = HoughLines(H,rhoRes,thetaRes,nLines)
        print("lRho: ", lRho, "\n")
        print("lTheta: ", lTheta, "\n")
        # return

        # l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()