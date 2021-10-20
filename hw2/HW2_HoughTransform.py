import math
import glob
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# parameters
datadir = './data'
resultdir='./results'
# you can calibrate these parameters
sigma=3
highThreshold=0.35
lowThreshold=0.25
rhoRes=2
thetaRes=math.pi/90
nLines=20

def replicatePadding(Igs, G):
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

# Returns bool whether target is max compared to elements in compareList
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

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # 1. Smooth Igs w/ Gaussian kernel w/ sigma
    # set arbitrary g_kernel for test
    g_kernel = gkern((3, 3), sigma, sigma)

    # Replicate pad Igs and Convolve w/ g_kernel
    smoothed = ConvFilter(Igs, g_kernel).clip(0, 1)

    # 2. Convolve smoothed img w/ sobel operators => Ix, Iy
    sobel_g = np.array([[1], [2], [1]])
    sobel_d = np.array([[-1, 0, 1]])
    Ix = ConvFilter(ConvFilter(smoothed, sobel_g), sobel_d) # 이건 clip 하면 안됨
    Iy = ConvFilter(ConvFilter(smoothed, sobel_g.T), sobel_d.T) # 이건 clip 하면 안됨
    # 3. Get gradient magnitude img, gradient direction img Io
    Imag = np.sqrt(np.add(np.square(Ix), np.square(Iy))).clip(0, 1)
    Io = np.arctan2(Iy, Ix)

    # 4. Apply non maximal suppression
    for x in range(Io.shape[0]):
        for y in range(Io.shape[1]):
            theta = Io[x][y]
            target_g = Imag[x][y]
            # 수평 방향
            if (-1/8)*math.pi <= theta <= (1/8)*math.pi or (theta >= (7/8)*math.pi) or (theta <= (-7/8)*math.pi) :
                l = Imag[x][y-1] if y>0 else 0
                r = Imag[x][y+1] if y<Io.shape[1]-1 else 0
                if not isMaxN(target_g, [l, r]):
                    Imag[x][y] = 0
            # 우상향 대각 방향
            elif (1/8)*math.pi < theta < (3/8)*math.pi or (-7/8)*math.pi < theta < (-5/8)*math.pi :
                ld = Imag[x+1][y-1] if y>0 and x<Io.shape[0]-1 else 0
                ru = Imag[x-1][y+1] if y<Io.shape[1]-1 and x>0 else 0
                if not isMaxN(target_g, [ld, ru]):
                    Imag[x][y] = 0
            # 수직 방향
            elif (3/8)*math.pi <= theta <= (5/8)*math.pi or (-5/8)*math.pi < theta < (-3/8)*math.pi:
                u = Imag[x-1][y] if x>0 else 0
                d = Imag[x+1][y] if x<Io.shape[0]-1 else 0
                if not isMaxN(target_g, [u, d]):
                    Imag[x][y] = 0
            # 좌상향 대각 방향
            else:
                lu = Imag[x-1][y-1] if x>0 and y>0 else 0
                rd = Imag[x+1][y+1] if x<Io.shape[0]-1 and y<Io.shape[1]-1 else 0
                if not isMaxN(target_g, [lu, rd]):
                    Imag[x][y] = 0

    # 5. Apply double thresholding
    for x in range(Imag.shape[0]):
        for y in range(Imag.shape[1]):
            target = Imag[x][y]
            if target >= highThreshold:
                continue
            elif target < lowThreshold:
                Imag[x][y] = 0
            else:
                neighbors = Imag[max(x-1, 0):min(Imag.shape[0], x+2), max(y-1, 0):min(Imag.shape[1], y+2)]
                if np.max(neighbors) < highThreshold:
                    Imag[x][y] = 0

    return Imag, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    rhoMax = math.sqrt((Im.shape[0])**2 + (Im.shape[1])**2) ## changed
    thetaMax = math.pi ## changed

    H = np.zeros((int(rhoMax/rhoRes)+1, int(thetaMax/thetaRes)+1))
    print("Shape of H: ", H.shape, "\n")

    # 4. For every non zero pixel in Im, calculate theta, rho -> Vote on H
    # Origin: bottom left
    for y in range(Im.shape[0]):
        y_from_origin = (Im.shape[0]-1) - y
        for x in range(Im.shape[1]):
            x_from_origin = x - Im.shape[1]/2
            if Im[y][x] != 0:
                theta = 0

                # add theta by theta Res every cycle
                while theta <= thetaMax:
                    rho = x_from_origin * np.cos(theta) + y_from_origin * np.sin(theta)
                    rho_normalized = math.ceil(rho / rhoRes)
                    theta_normalized = math.ceil(theta / thetaRes)
                    H[rho_normalized][theta_normalized] += 1
                    theta += thetaRes

    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # 1. Init returned list
    lRho = []
    lTheta = []
    # 2. Non maximal suppression
    for y in range(H.shape[1]):
        for x in range(H.shape[0]):

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
        print("Top", i, ": theta", normal_theta, " rho: ", normal_rho, "\n")
        print("Received votes: ", H[normal_rho][normal_theta], "\n")
        print("MAX: ", H.max(), "\n")
        H[normal_rho][normal_theta] = 0
    # 4. Restore original rho, theta value
    lRho = [rho * rhoRes for rho in lRho]
    lTheta = [theta * thetaRes for theta in lTheta]
    return lRho, lTheta

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
def getY(slope, y_intercept, x):
    y = int(slope * x + y_intercept)
    return y
def getX(slope, y_intercept, y):
    x = int((y - y_intercept)/slope)
    return x
def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        im = Image.open(img_path).convert("L")
        #
        # img = np.zeros((200, 200))
        # img[100] = 255
        # img[:,100] = 255
        # for i in range(80):
        #     img[199-i][119+i] = 255
        #     img[150-i][i] = 255
        #     img[119+i][i] = 255

        plt.figure()
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.show()
        Igs = np.array(im)
        Igs = Igs / 255.
        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)

        # # temp test
        # Im = np.zeros((200, 200))
        # Im[100] = 1
        # Im[:,100] = 1
        # for i in range(80):
        #     Im[199-i][119+i] = 1
        #     Im[150-i][i] = 1
        #     Im[119+i][i] = 1

        H= HoughTransform(Im, rhoRes, thetaRes)
        # temp
        print("Hough:\n")
        plt.figure()
        plt.imshow(H, cmap='gray')
        plt.show()

        lRho,lTheta = HoughLines(H,rhoRes,thetaRes,nLines)
        print("lRho: ", lRho, "\n")
        print("lTheta: ", lTheta, "\n")
        # return

        # plot hough lines on org img
        im = Image.open(img_path)
        draw = ImageDraw.Draw(im)

        # return
        for i in range(nLines):
            print("Doing ", i, "\n")
            shape = []

            imgMaxX = Igs.shape[1]-1
            imgMaxY = Igs.shape[0]-1

            x_origin = Igs.shape[1]/2

            # theta = 0 -> Cx + rho 에 수직
            if lTheta[i] == 0:
                shape = [(x_origin + lRho[i], 0), (x_origin + lRho[i], Igs.shape[0])]
            # theta = pi -> Cx - rho 에 수직
            elif lTheta[i] == math.pi:
                shape = [(x_origin - lRho[i], 0), (x_origin - lRho[i], Igs.shape[0])]
            else:
                # get slope
                slope = - np.cos(lTheta[i]) / np.sin(lTheta[i])
                y_intercept = lRho[i] / np.sin(lTheta[i])

                print("Slope: ", slope, " intercept: ", y_intercept, "\n")
                # if y is valid in x = 0
                if 0 <= getY(slope, y_intercept, 0) <= imgMaxY:
                    shape.append((0, imgMaxY - getY(slope, y_intercept, 0)))
                # if y is valid in imgMaxX
                if 0 <= getY(slope, y_intercept, imgMaxX) <= imgMaxY:
                    shape.append((imgMaxX, imgMaxY - getY(slope, y_intercept, imgMaxX)))
                # if x is valid in 0
                if 0 <= getX(slope, y_intercept, 0) <= imgMaxX:
                    shape.append((getX(slope, y_intercept, 0), imgMaxY))
                # if x is valid in imgMaxY
                if 0 <= getX(slope, y_intercept, imgMaxY) <= imgMaxX:
                    shape.append((getX(slope, y_intercept, imgMaxY), 0))

            # draw line
            draw.line(shape, width=1, fill='#00ff00')
            # draw.line([(0,0), (Igs.shape[1]-1, Igs.shape[0]-1)], width=1, fill='red')

        im.show()
        # l = HoughLineSegments(lRho, lTheta, Im)
        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
if __name__ == '__main__':
    main()