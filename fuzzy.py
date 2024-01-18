import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

min_threshold = 88
max_threshold = 64

# p1 p2 p3
# p4 p5 p6
# p7 p8 p9

ruleset3x3 = {
    "11111000": "e",
    "00011111": "e",
    "01101011": "e",
    "11010110": "e",
    "00101011": "e",
    "11010100": "e",
    "01101001": "e",
    "10010110": "e",
    "00001111": "e",
    "11101000": "e",
    "00010111": "e",
    "11110000": "e",
    "00000111": "e",
    "00101001": "e",
    "11100000": "e",
    "10010100": "e",
    "01101111": "e",
    "11101011": "e",
    "11111001": "e",
    "00111111": "e",
    "11111100": "e",
    "11110110": "e",
    "11010111": "e",
    "10011111": "e",
    "10010111": "e",
    "11110100": "e",
    "00101111": "e",
    "11101001": "e",
}


ruleset = {
    "b0": [0, 0, 0, 0],
    "e1": [0, 0, 0, 1],
    "e2": [0, 0, 1, 0],
    "e2": [0, 0, 1, 1],
    "e3": [0, 1, 0, 0],
    "e4": [0, 1, 0, 1],
    "e5": [0, 1, 1, 0],
    "w0": [0, 1, 1, 1],
    "e6": [1, 0, 0, 0],
    "e7": [1, 0, 0, 1],
    "e8": [1, 0, 1, 0],
    "w1": [1, 0, 1, 1],
    "e9": [1, 1, 0, 0],
    "w2": [1, 1, 0, 1],
    "w3": [1, 1, 1, 0],
    "w4": [1, 1, 1, 1]
}

def check_ruleset2x2(p1, p2, p3, p4):
        mask = [p1, p2, p3, p4]
        for key in ruleset:
            if (ruleset[key] == mask):
                return key[0]

def check_ruleset(p1, p2, p3, p4, p6, p7, p8, p9):
    neighbors = concatenate_numbers(p1, p2, p3, p4, p6, p7, p8, p9)
    if (neighbors in ruleset3x3):
        return 'e'
    else:
        return 'b'

def thold(value, th):
    if value <= th:
        return 0
    else:
        return 1
    
def concatenate_numbers(num1, num2, num3, num4, num5, num6, num7, num8):
    concatenated_string = str(num1) + str(num2) + str(num3) + str(num4) + str(num5) + str(num6) + str(num7) + str(num8)
    return concatenated_string

def fuzzy2x2(image):
    result_image = np.zeros_like(image)

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):

            p1 = thold(image[i - 1, j - 1])
            p2 = thold(image[i - 1, j])
            p3 = thold(image[i , j - 1 ])
            p4 = thold(image[i, j])

            membership = check_ruleset2x2(p1, p2, p3, p4)
            if (membership == 'e'):
                result_image[i, j] = 255
            elif (membership == 'w'):
                result_image[i, j] = 10
            elif (membership == 'b'):
                result_image[i, j] = 0
    return result_image

def fuzzy3x3(image, mean):
    result_image = np.zeros_like(image)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):

            p1 = (image[i - 1, j - 1])
            p2 = (image[i - 1, j])
            p3 = (image[i - 1, j + 1])
            p4 = (image[i, j - 1])
            p5 = image[i, j]
            p6 = (image[i, j + 1])
            p7 = (image[i + 1, j - 1])
            p8 = (image[i + 1, j])
            p9 = (image[i + 1, j + 1])
            # avg = round((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9)
            # print(avg)
            p1 = thold(p1, mean)
            p2 = thold(p2, mean)
            p3 = thold(p3, mean)
            p4 = thold(p4, mean)
            p6 = thold(p6, mean)
            p7 = thold(p7, mean)
            p8 = thold(p8, mean)
            p9 = thold(p9, mean)


            membership = check_ruleset(p1, p2, p3, p4, p6, p7, p8, p9)
            if (membership == 'e'):
                result_image[i, j] = 255
            elif (membership == 'w'):
                result_image[i, j] = 0
            elif (membership == 'b'):
                result_image[i, j] = 0
    return result_image

filenames = os.listdir('train')
# print(filenames)

for file in filenames:

    print(file)

    in_filename = 'train/' + file
    out_filename = 'f_results/' + file
    # Load an example image
    image = cv2.imread(in_filename, cv2.IMREAD_GRAYSCALE)

    men = np.mean(image)
    print(men)

    # p1 p2 p3
    # p4 p5 p6
    # p7 p8 p9

    # Apply fuzzy logic to the input image
    result_image = fuzzy3x3(image, men)

    # # Display the original and result images
    # plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    # plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title('Fuzzy Edge Detection')
    # plt.show()
    cv2.imwrite(out_filename, result_image)