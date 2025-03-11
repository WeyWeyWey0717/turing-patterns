import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal

# # 
# data = np.loadtxt('output_files/grid_data_256_10000000_step_c=0_24.txt').astype(int)
# data = data.flatten()
# # data = np.loadtxt('20241211_grid_data_64_10000000_step_c=0_25_alpha=_1.txt').astype(int)
# # print(data.shape)
# # 'grid_data_256_10000000_step_c=0_24.txt': 0 is more 
# count_ones = np.count_nonzero(data == 1)
# print(f"Number of 1s in data: {count_ones}")
# count_zeros = np.count_nonzero(data == 0)
# print(f"Number of 0s in data: {count_zeros}")



# grid_size = 256
# x = np.arange(grid_size)
# y = np.arange(grid_size)
# X, Y = np.meshgrid(x, y)

# interval = [1, 1.5, 1.75, 2]

# X_hex = X + 0.5 * Y
# Y_hex = (np.sqrt(3)/2) * Y
# alpha = interval[0]
# Y_hex = alpha * Y_hex

# # Number of points
# num_points = X.size  # 256*256 = 65536

# # Plot the points colored by their value
# X_flat = X_hex.flatten()
# Y_flat = Y_hex.flatten()

# color_map = ['red' if v == 0 else 'blue' for v in data]

# plt.figure(figsize=(6,6))
# markersize = 3
# plt.scatter(X_flat, Y_flat, c=color_map, marker='H', linewidth=markersize)
# plt.title('Coordinates with Assigned Values (0 or 1)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()










# 20241212_grid_data_256_100000_step_c=0_33_alpha=_1_value1_coordinates
# data = np.loadtxt('../turing-patterns/20241212_MC_Results/20241212_grid_data_256_100000_step_c=0_25_alpha=_1.75_value1_coordinates.txt')
data = np.loadtxt('../turing-patterns/20241212_MC_Results/20241212_grid_data_256_100000_step_c=0_33_alpha=_1.25_value1_coordinates.txt')
print(data.shape)
plt.figure(figsize=(6,6))
markersize = 3
plt.scatter(data[:,0], data[:,1], c='blue', marker='H', linewidth=markersize)
plt.title('Coordinates with Assigned Values (0 or 1)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 500)
y = np.linspace(np.min(data[:,1]), np.max(data[:,1]), 500)
X, Y = np.meshgrid(x, y)

# Define the Gaussian function
# def gaussian(x, y, sigma):
#     return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Apply the Gaussian function to each point
sigma = 0.5  # You can adjust the sigma value as needed

position_map_gaussian = 0
for i in range(len(data)): #len(data)
    position_map_gaussian += np.exp(-((X - data[i,0])**2 + (Y - data[i,1])**2) / (2 * sigma**2))



# Plot the heatmap
fig, axs = plt.subplots(1, 2, figsize=(10,6))
axs[0].imshow(np.flipud(position_map_gaussian), cmap='viridis', extent=[np.min(data[:,0]), np.max(data[:,0]), np.min(data[:,1]), np.max(data[:,1])])
axs[0].set_title('Heatmap of Gaussian Distribution')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Perform FFT on the Gaussian position map
fft_map = np.fft.fftshift(np.fft.fft2(position_map_gaussian))

# Plot the FFT result
fig01 = axs[1].imshow(np.log(np.abs(fft_map)), cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5])
axs[1].set_title('FFT of Gaussian Distribution')
axs[1].set_xlabel('Frequency X')
axs[1].set_ylabel('Frequency Y')
plt.colorbar(fig01)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

