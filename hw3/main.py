import math
import numpy as np
from PIL import Image

# @ param p correspondence coordinates array e.g., [(173,10), (222,319), ... ]
# @ retval ret normalized correspondence coordinates array
# @ retval T transformation matrix used for p normalization
def normalize(p):
    print("p: ", p, "\n")
    print("p.shape: ", p.shape, "\n")
    ret = np.zeros(p.shape)
    centroid_x, centroid_y = np.mean(p, axis=0)

    # 1. Translate centroid to origin
    T = np.array([[1, 0, -centroid_x], [0, 1, -centroid_y], [0, 0, 1]])
    print("T1 for translation: ", T, "\n")

    # 2. Scale such that avg distance from origin == math.sqrt(2)
    sum_distance = 0
    for pt in p:
        sum_distance += math.sqrt(pt[0]**2 + pt[1]**2)
    avg_distance = sum_distance / p.shape[0]
    print("avg_distance: ", avg_distance, "\n")
    scale_factor = avg_distance / math.sqrt(2)
    print("scale factor: ", scale_factor, "\n")

    T /= scale_factor
    print("T1 after scaling: ", T, "\n")

    # 3. Normalize pts in p
    for i in range(p.shape[0]):
        pt_i = np.array([p[i][0], p[i][1], 1]).T
        pt_i_prime = np.dot(T, pt_i)
        pt_normalized = np.array([pt_i_prime[0]/pt_i_prime[2], pt_i_prime[1]/pt_i_prime[2]]).T
        ret[i] = pt_normalized

    return ret, T

def compute_h(p1, p2):
    n = p1.shape[0]
    A = np.zeros((2*n, 9), dtype=np.int64)

    for i in range(n):
        xi, yi = p2[i][0], p2[i][1]
        xi_p, yi_p = p1[i][0], p1[i][1]

        A[2*i] = [xi, yi, 1, 0, 0, 0, -xi_p * xi, -xi_p*yi, -xi_p]
        A[2*i+1] = [0, 0, 0, xi, yi, 1, -yi_p*xi, -yi_p*yi, -yi_p]

    U, D, Vt = np.linalg.svd(A)
    V = Vt.T
    H = np.reshape(V[:, np.argmin(D)], (3, 3))

    return H

def compute_h_norm(p1, p2):
    if p1.shape[0] != p2.shape[0]:
        print("Correspondence lists differ in shape\n")
        return

    p1_normal, T1 = normalize(p1)
    p2_normal, T2 = normalize(p2)

    print("p1 normal: ", p1_normal, "\n")
    print("p2 normal: ", p2_normal, "\n")

    H_hat = compute_h(p1_normal, p2_normal)
    T1_inv = np.linalg.inv(T1)
    H = np.dot(np.dot(T1_inv, H_hat), T2)

    return H


def compute_h_norm2(p1, p2):
    # TODO ...
    # number of pairs of corresponding interest points
    N = p1.shape[0]

    # construct Normalization matrix T1, T2
    p2_mean = np.mean(p2, axis=0)
    p1_mean = np.mean(p1, axis=0)
    sump2 = 0
    sump1 = 0
    for i in range(N):
        sump2 += ((p2[i][0] - p2_mean[0]) ** 2 + (p2[i][1] - p2_mean[1]) ** 2) ** 0.5
        sump1 += ((p1[i][0] - p1_mean[0]) ** 2 + (p1[i][1] - p1_mean[1]) ** 2) ** 0.5
    s2 = (math.sqrt(2) * N) / sump2
    s1 = (math.sqrt(2) * N) / sump1
    T2 = s2 * np.array([[1, 0, -p2_mean[1]], [0, 1, -p2_mean[0]], [0, 0, 1 / s2]])
    T1 = s1 * np.array([[1, 0, -p1_mean[1]], [0, 1, -p1_mean[0]], [0, 0, 1 / s1]])

    # normalize p1, p2 using T1, T2
    p2_norm = np.zeros(p2.shape)
    p1_norm = np.zeros(p1.shape)
    for i in range(N):
        coord2 = np.array([p2[i][1], p2[i][0], 1])
        result2 = np.matmul(T2, coord2.T)
        p2_norm[i][0] = result2[1]
        p2_norm[i][1] = result2[0]

        coord1 = np.array([p1[i][1], p1[i][0], 1])
        result1 = np.matmul(T1, coord1.T)
        p1_norm[i][0] = result1[1]
        p1_norm[i][1] = result1[0]

    # construct A
    A = [-p2_norm[0][1], -p2_norm[0][0], -1, 0, 0, 0, p1_norm[0][1] * p2_norm[0][1], p1_norm[0][1] * p2_norm[0][0],
         p1_norm[0][1]]
    for i in range(N):
        arr1 = [-p2_norm[i][1], -p2_norm[i][0], -1, 0, 0, 0, p1_norm[i][1] * p2_norm[i][1],
                p1_norm[i][1] * p2_norm[i][0], p1_norm[i][1]]
        arr2 = [0, 0, 0, -p2_norm[i][1], -p2_norm[i][0], -1, p1_norm[i][0] * p2_norm[i][1],
                p1_norm[i][0] * p2_norm[i][0], p1_norm[i][0]]
        if i != 0:
            A = np.vstack((A, arr1))
        A = np.vstack((A, arr2))

    # apply SVD
    U, S, Vh = np.linalg.svd(A)
    V = Vh.T
    minev = V[:, np.argmin(S)]
    H_bar = np.reshape(minev, (3, 3))

    # undo normalzation
    T1_inv = np.linalg.inv(T1)
    mid = np.matmul(T1_inv, H_bar)
    H = np.matmul(mid, T2)

    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...

    return igs_rec

def set_cor_mosaic():
    # TODO ...

    return p_in, p_ref

def set_cor_rec():
    # TODO ...

    return c_in, c_ref

def main():
    p1 = np.array([(125, 187), (175, 187), (175, 224), (125, 225)])
    p2 = np.array([(134, 210), (192, 214), (191, 236), (135, 233)])

    H = compute_h(p1, p2)
    H_norm = compute_h_norm(p1, p2)
    H_norm2 = compute_h_norm2(p1, p2)

    print("*"*30, "\n")
    print("H: ", H, "\n")
    print("H Normalized: ", H_norm, "\n")
    print("H Normalized2: ", H_norm2, "\n")
    print("*"*30, "\n")

    #
    # ##############
    # # step 1: mosaicing
    # ##############
    #
    # # read images
    # img_in = Image.open('data/porto1.png').convert('RGB')
    # img_ref = Image.open('data/porto2.png').convert('RGB')
    #
    # # shape of igs_in, igs_ref: [y, x, 3]
    # igs_in = np.array(img_in)
    # igs_ref = np.array(img_ref)
    #
    # # lists of the corresponding points (x,y)
    # # shape of p_in, p_ref: [N, 2]
    # p_in, p_ref = set_cor_mosaic()
    #
    # # p_ref = H * p_in
    # H = compute_h_norm(p_ref, p_in)
    # igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)
    #
    # # plot images
    # img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    # img_merge = Image.fromarray(igs_merge.astype(np.uint8))
    #
    # # save images
    # img_warp.save('porto1_warped.png')
    # img_merge.save('porto_mergeed.png')
    #
    # ##############
    # # step 2: rectification
    # ##############
    #
    # img_rec = Image.open('data/iphone.png').convert('RGB')
    # igs_rec = np.array(img_rec)
    #
    # c_in, c_ref = set_cor_rec()
    #
    # igs_rec = rectify(igs_rec, c_in, c_ref)
    #
    # img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    # img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
