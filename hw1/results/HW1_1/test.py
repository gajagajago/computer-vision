import numpy as np

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

if __name__ == '__main__':
    a = np.arange(25).reshape(5,5)
    b = pad_reflect(a, 2)

