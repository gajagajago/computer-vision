import numpy as np

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

if __name__ == '__main__':
    a = np.arange(30).reshape(5,-1)
    b = pad_reflect(a, (2,1))
    print(b)

