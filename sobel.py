from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2

# input images
image_file = 'image.jpg'
# image_file = 'image2.jpg'
# image_file = 'image3.jpg'

def rgb2grayscale(img: np.ndarray):
    """
    img: image in RGB format
    return: image in grayscale format
    """
    return np.around((img[:, :, 0]*0.3 + img[:, :, 1]*0.59 + img[:, :, 2]*0.11))

def sobel():
	input_image = cv2.imread(image_file)  # this is the array representation of the input image
	[nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB)

	grayscale_image = input_image.copy()
    
    # Convert to grayscale (in case is RGB)
	if len(input_image.shape) == 3:
		grayscale_image = rgb2grayscale(img=grayscale_image)


	# Here we define the matrices associated with the Sobel filter
	Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
	Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
	[rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image

	sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)
	# Now we "sweep" the image in both x and y directions and compute the output
	for i in range(rows - 2):
		for j in range(columns - 2):
			gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
			gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
			sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

	# Display the original image and the Sobel filtered image
	fig2 = plt.figure(1)
	ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
	# BGR -> RGB
	ax1.imshow(input_image[:,:,::-1])
	ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
	fig2.show()

	plt.show()

	# Save the filtered image in destination path
	plt.imsave('results/sobel_resultorig.png', sobel_filtered_image, cmap=plt.get_cmap('gray'))

	return sobel_filtered_image

sobel()
