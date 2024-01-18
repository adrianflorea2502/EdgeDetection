import numpy as np

def to_gray(img: np.ndarray, format: str):
    '''
    Algorithm:
    >>> 0.2989 * R + 0.5870 * G + 0.1140 * B 
    - Returns a gray image
    '''

    r_coef = 0.2989
    g_coef = 0.5870
    b_coef = 0.1140

    if format.lower() == 'bgr':
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        return r_coef * r + g_coef * g + b_coef * b
    elif format.lower() == 'rgb':
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return r_coef * r + g_coef * g + b_coef * b
    else:
        raise Exception('Unsupported value in parameter \'format\'')