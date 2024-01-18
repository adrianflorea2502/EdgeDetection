from sobel import sobel
from canny import canny
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sobel_img, canny_img = canny()

sob_x, sob_y = sobel_img.shape
can_x, can_y = canny_img.shape

if (sob_x > can_x):
    sobel_img = sobel_img[0:can_x, 0:sob_y]
    sob_x = can_x
else:
    canny_img = canny_img[0:sob_x, 0:can_y]
    can_x = sob_x

if (sob_y > can_y):
    sobel_img = sobel_img[0:sob_x, 0:can_y]
    sob_y = can_y
else:
    canny_img = canny_img[0:can_x, 0:sob_y]
    can_y = sob_y

combined_image = 0.5 * sobel_img + 0.5 * canny_img

# combined_image = combined_image.swapaxes(0,1)
# combined_image = combined_image.swapaxes(1,2)

imgplot = plt.imshow(combined_image, cmap='gray')
plt.show()
cv2.imwrite('results/hybrid_result.png', combined_image)

# r= 0.5 * sobel_img + 0.5 * canny_img
# rr=plt.imshow(r, cmap='gray')
# plt.show()
