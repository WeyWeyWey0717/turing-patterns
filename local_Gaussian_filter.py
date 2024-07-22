import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import peak_local_max


# Load the image
image_path = './c=0_24_simulation.png'
image = mpimg.imread(image_path)
print(image.shape) # 1290, 1282, 4
# RGBsum_image = np.zeros((1290, 1282))

# if abs(np.sum(image[:,:,0:2]) - 3) < 0.1:
#     image[:,:,0:2] = 0
# RGBsum_image[:,:] = -np.sqrt(image[:,:,0]**2 + image[:,:,1]**2 + image[:,:,2]**2)
# print(RGBsum_image.shape)
# peaks_coordinates = peak_local_max(RGBsum_image, min_distance=10)

# # Plotting the peak coordinates
# plt.figure()
# plt.imshow(RGBsum_image, cmap='gray')
# plt.plot(peaks_coordinates[:, 1], peaks_coordinates[:, 0], 'r.')
# plt.title('Peak Coordinates')
# plt.xlabel('x (pix)')
# plt.ylabel('y (pix)')
# plt.show()



# # Parameters for the local Gaussian filter
# block_size = 25  # Size of each block
# sigma = 2.5  # Standard deviation for Gaussian kernel within each block

# # Get the dimensions of the image
# height, width, channels = image.shape

# # Create a copy of the image to store the filtered result
# smoothed_image_local = np.copy(image)

# # Apply local Gaussian filtering block by block
# for i in range(0, height, block_size):
#     for j in range(0, width, block_size):
#         # Define the boundaries of the current block
#         block = image[i:i+block_size, j:j+block_size]
#         # Apply Gaussian filter to the current block
#         smoothed_block = gaussian_filter(block, sigma=sigma)
#         # Place the smoothed block back into the result image
#         smoothed_image_local[i:i+block_size, j:j+block_size] = smoothed_block


coordinates = []
for i in range(38):
    for j in range(69):
        coordinates.append([10+i*34.59, 9+j*18.59]) # estimate: 1282/38 and 1290/69
for i in range(37):
    for j in range(69):
        coordinates.append([27+i*34.59, 0+j*18.59])
coordinates = np.array(coordinates)

sigma = 1 # Standard deviation for Gaussian kernel

# Create a copy of the image to store the filtered result
smoothed_image_local = np.copy(image)

block_size = 11
# Apply the Gaussian filter centered at each (x, y) coordinate
for cx, cy in coordinates:
    # Define the boundaries of the current block
    block = image[int(cx-block_size):int(cx+block_size), int(cy-block_size):int(cy+block_size)]
    # Apply Gaussian filter to the current block
    smoothed_block = gaussian_filter(block, sigma=sigma)
    # Place the smoothed block back into the result image
    smoothed_image_local[int(cx-block_size):int(cx+block_size), int(cy-block_size):int(cy+block_size)] = smoothed_block


# Display the original and locally smoothed images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
# ax[0].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[0].set_title('Original Image')
ax[0].axis('off')
print(image.shape)

ax[1].imshow(smoothed_image_local)
ax[1].set_title('Locally Smoothed Image')
ax[1].axis('off')
# print(smoothed_image_local.shape)

plt.show()







# def square_to_hex(values):
#     # # Define values for each point in the grid
#     # values = np.fliplr(data)

#     # Define original coordinates of the grid
#     pix_x = values.shape[0]
#     pix_y = values.shape[1]
#     x = np.arange(0,pix_x)  # Original x-coordinates
#     y = np.arange(0,pix_y)  # Original y-coordinates
#     xx, yy = np.meshgrid(x, y)  # Create a grid of original coordinates

#     # Stack the original coordinates and values into a single array
#     original_points = np.dstack([xx, yy, values])

#     # Define transformation matrix for making hexagonal grid
#     # theta = np.pi / 6  # Angle of rotation for hexagonal grid
#     # s = 1 / np.cos(theta)  # Scale factor to maintain spacing
#     s = 1
#     transformation_matrix = np.array([[s * 1, s * 1/2],
#                                     [s * 0, s * np.sqrt(3)/2]])

#     # Apply the transformation to all points in the grid (excluding the values)
#     transformed_points = np.dot(original_points[:, :, :2].reshape(-1, 2), transformation_matrix.T)
#     transformed_points = transformed_points.reshape(original_points.shape[0], original_points.shape[1], -1)

#     # Extract transformed coordinates and values
#     transformed_coordinates = transformed_points[:, :, :2]
#     values = original_points[:, :, 2]

#     return transformed_coordinates, values

# # Load the data from the file
# data = np.loadtxt('grid_data_256_10000000_step_c=0_24.txt').astype(float)

# # Plot the data
# trans_data, values = square_to_hex(data)
# # plt.scatter(trans_data[:,:,0],np.flipud(trans_data[:,:,1]), c=(-1)*(values-1), cmap='viridis', linewidths=0.2) # marker='H',
# # # trans_data[:,:,0],np.flipud(trans_data[:,:,1]) are the "(x,y) coordinates" of the hexagons
# # plt.title('Data') 
# # plt.show()

# print(values.shape)
# sigma = 10  # Standard deviation for Gaussian kernel
# filter_size = 11  # Size of the filter (should be odd)

# # Create a Gaussian filter
# x = np.linspace(-filter_size//2, filter_size//2, filter_size)
# y = np.linspace(-filter_size//2, filter_size//2, filter_size)
# x, y = np.meshgrid(x, y)
# d = np.sqrt(x*x + y*y)
# gaussian_filter_mask = np.exp(-(d**2 / (2.0 * sigma**2)))

# # Normalize the filter
# gaussian_filter_mask /= gaussian_filter_mask.max()
# print(gaussian_filter_mask)

# # Create a copy of the image to store the filtered result
# smoothed_image = np.copy(values)

# coordinates = np.array([[50, 100], [150, 150], [200, 200]]) # [(trans_data[:,:,0], np.flipud(trans_data[:,:,1]))]

# # Apply the Gaussian filter centered at each (x, y) coordinate
# for cx, cy in coordinates:
#     # Define the boundaries of the filter area
#     xmin = max(cx - filter_size//2, 0)
#     xmax = min(cx + filter_size//2 + 1, values.shape[1])
#     ymin = max(cy - filter_size//2, 0)
#     ymax = min(cy + filter_size//2 + 1, values.shape[0])
    
#     # Extract the region of interest
#     roi = values[ymin:ymax, xmin:xmax]
    
#     # # Apply the Gaussian filter to the region of interest
#     # for channel in range(roi.shape[2]):
#     #     roi[..., channel] = np.multiply(roi[..., channel], gaussian_filter_mask[:roi.shape[0], :roi.shape[1]])
    
#     # Place the smoothed region back into the image
#     smoothed_image[ymin:ymax, xmin:xmax] = gaussian_filter_mask * roi

# # Display the original and smoothed images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(values)
# ax[0].set_title('Original Image')
# # ax[0].axis('off')

# ax[1].imshow(smoothed_image)
# ax[1].set_title('Image with Gaussian Filters at Specified Coordinates')
# # ax[1].axis('off')

# plt.show()