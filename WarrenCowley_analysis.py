import numpy as np
import matplotlib.pyplot as plt
import random

def square_to_hex(values):
    # # Define values for each point in the grid
    # values = np.fliplr(data)

    # Define original coordinates of the grid
    pix_x = values.shape[0]
    pix_y = values.shape[1]
    x = np.arange(0,pix_x)  # Original x-coordinates
    y = np.arange(0,pix_y)  # Original y-coordinates
    xx, yy = np.meshgrid(x, y)  # Create a grid of original coordinates

    # Stack the original coordinates and values into a single array
    original_points = np.dstack([xx, yy, values])

    # Define transformation matrix for making hexagonal grid
    # theta = np.pi / 6  # Angle of rotation for hexagonal grid
    # s = 1 / np.cos(theta)  # Scale factor to maintain spacing
    s = 1
    transformation_matrix = np.array([[s * 1, -s * 1/2],
                                    [s * 0, s * np.sqrt(3)/2]])

    # Apply the transformation to all points in the grid (excluding the values)
    transformed_points = np.dot(original_points[:, :, :2].reshape(-1, 2), transformation_matrix.T)
    transformed_points = transformed_points.reshape(original_points.shape[0], original_points.shape[1], -1)

    # Extract transformed coordinates and values
    transformed_coordinates = transformed_points[:, :, :2]
    values = original_points[:, :, 2]

    return transformed_coordinates, values



# Load the data from the file
data = np.loadtxt('grid_data_256_1e6_step.txt').astype(int)


neighbor_distance = [1, 3, 4, 7, 9] # in square unit
neighbor_numbers = [6, 6, 6, 12, 6]

# Function to get nearest neighbors with periodic boundary conditions
def get_neighbors(grid, row, col,):
    grid_size=grid.shape[0]
    neighbors = {'first': [], 'second': [], 'third': [], 'fourth': [], 'fifth': []}
    neighbors_coordinates = {'first': [], 'second': [], 'third': [], 'fourth': [], 'fifth': []}
    # temp = []
    for r in range(row-3, row+4):
        for c in range(col-3, col+4):
            if abs(((c-col)+1/2*(r-row))**2 + ((r-row)*np.sqrt(3)/2)**2 - neighbor_distance[0]) < 0.01:
                neighbors['first'].append(grid[r % grid_size, c % grid_size])
                neighbors_coordinates['first'].append([r % grid_size, c % grid_size])
            if abs(((c-col)+1/2*(r-row))**2 + ((r-row)*np.sqrt(3)/2)**2 - neighbor_distance[1]) < 0.01:
                neighbors['second'].append(grid[r % grid_size, c % grid_size])
                neighbors_coordinates['second'].append([r % grid_size, c % grid_size])
            if abs(((c-col)+1/2*(r-row))**2 + ((r-row)*np.sqrt(3)/2)**2 - neighbor_distance[2]) < 0.01:
                neighbors['third'].append(grid[r % grid_size, c % grid_size])
                neighbors_coordinates['third'].append([r % grid_size, c % grid_size])
            if abs(((c-col)+1/2*(r-row))**2 + ((r-row)*np.sqrt(3)/2)**2 - neighbor_distance[3]) < 0.01:
                neighbors['fourth'].append(grid[r % grid_size, c % grid_size])
                neighbors_coordinates['fourth'].append([r % grid_size, c % grid_size])
            if abs(((c-col)+1/2*(r-row))**2 + ((r-row)*np.sqrt(3)/2)**2 - neighbor_distance[4]) < 0.01:
                neighbors['fifth'].append(grid[r % grid_size, c % grid_size])
                neighbors_coordinates['fifth'].append([r % grid_size, c % grid_size])
    # temp for printing the exact coordinates of the neighbors
    #             temp.append([r,c])
    # print(temp)
    return neighbors, neighbors_coordinates

def square_to_hex(values):
    # # Define values for each point in the grid
    # values = np.fliplr(data)

    # Define original coordinates of the grid
    pix_x = values.shape[0]
    pix_y = values.shape[1]
    x = np.arange(0,pix_x)  # Original x-coordinates
    y = np.arange(0,pix_y)  # Original y-coordinates
    xx, yy = np.meshgrid(x, y)  # Create a grid of original coordinates

    # Stack the original coordinates and values into a single array
    original_points = np.dstack([xx, yy, values])

    # Define transformation matrix for making hexagonal grid
    # theta = np.pi / 6  # Angle of rotation for hexagonal grid
    # s = 1 / np.cos(theta)  # Scale factor to maintain spacing
    s = 1
    transformation_matrix = np.array([[s * 1, s * 1/2],
                                    [s * 0, s * np.sqrt(3)/2]])

    # Apply the transformation to all points in the grid (excluding the values)
    transformed_points = np.dot(original_points[:, :, :2].reshape(-1, 2), transformation_matrix.T)
    transformed_points = transformed_points.reshape(original_points.shape[0], original_points.shape[1], -1)

    # Extract transformed coordinates and values
    transformed_coordinates = transformed_points[:, :, :2]
    values = original_points[:, :, 2]

    return transformed_coordinates, values




search_zero_find_one = []
for i in range(100000):
    # Find a random "one" and "zero"
    ones_indices = np.argwhere(data == 1)
    zeros_indices = np.argwhere(data == 0)

    random_one = random.choice(ones_indices)
    random_zero = random.choice(zeros_indices)

    # Get neighbors for the random "one" and "zero"
    one_neighbors, one_neighbors_coordinates = get_neighbors(data, random_one[0], random_one[1])
    zero_neighbors, zero_neighbors_coordinates = get_neighbors(data, random_zero[0], random_zero[1])
    # print(type(one_neighbors['first']))

    # # FOR DEBUGGING
    # if i == 0:
        # nth_neighbor = 'fifth'
        # print([random_one[0], random_one[1]], one_neighbors[nth_neighbor], one_neighbors_coordinates[nth_neighbor])
        # axs[0,0].scatter(random_one[1], random_one[0])
        # axs[0,0].scatter(np.array(one_neighbors_coordinates[nth_neighbor])[:,1], np.array(one_neighbors_coordinates[nth_neighbor])[:,0])
        # print([random_zero[0], random_zero[1]], zero_neighbors[nth_neighbor], zero_neighbors_coordinates[nth_neighbor])
        # axs[0,0].scatter(random_zero[1], random_zero[0])
        # axs[0,0].scatter(np.array(zero_neighbors_coordinates[nth_neighbor])[:,1], np.array(zero_neighbors_coordinates[nth_neighbor])[:,0])

    
    # counting the status that is different from the center
    one_neighbor_zero = np.array([one_neighbors['first'].count(0), one_neighbors['second'].count(0), one_neighbors['third'].count(0), one_neighbors['fourth'].count(0), one_neighbors['fifth'].count(0)])
    one_neighbor_one = np.array([one_neighbors['first'].count(1), one_neighbors['second'].count(1), one_neighbors['third'].count(1), one_neighbors['fourth'].count(1), one_neighbors['fifth'].count(1)])
    zero_neighbor_one = np.array([zero_neighbors['first'].count(1), zero_neighbors['second'].count(1), zero_neighbors['third'].count(1), zero_neighbors['fourth'].count(1), zero_neighbors['fifth'].count(1)])
    zero_neighbor_zero = np.array([zero_neighbors['first'].count(0), zero_neighbors['second'].count(0), zero_neighbors['third'].count(0), zero_neighbors['fourth'].count(0), zero_neighbors['fifth'].count(0)])

    # print(one_neighbor_one, one_neighbor_zero)
    search_zero_find_one.append(zero_neighbor_one)

    # e_before = one_count + zero_count
    # e_after = 12 - e_before

    # # Flip the random "one" and "zero" if energy decreases
    # if e_after > e_before:    
    #     grid[random_one[0], random_one[1]] = 0
    #     grid[random_zero[0], random_zero[1]] = 1
    # else:
    #     continue


    # # Display results
    # print(f"Random one at {random_one}: {grid[random_one[0], random_one[1]]}")
    # print(f"Neighbors of the random one: {one_neighbors}")
    # print(f"Number of zeros around the random one: {one_count}")

    # print(f"Random zero at {random_zero}: {grid[random_zero[0], random_zero[1]]}")
    # print(f"Neighbors of the random zero: {zero_neighbors}")
    # print(f"Number of ones around the random zero: {zero_count}")
search_zero_find_one = np.array(search_zero_find_one)
print(search_zero_find_one.shape)
probability = []
probability_std = []
for i in range(search_zero_find_one.shape[1]):
    probability.append(search_zero_find_one[:,i].mean()/neighbor_numbers[i])
    probability_std.append(search_zero_find_one[:,i].std()/neighbor_numbers[i])
probability = np.array(probability)
print(probability)
print(probability_std)

# Plot the probability and its standard deviation
fig, axs = plt.subplots(2, 3)
x = np.arange(len(probability))
axs[0,0].errorbar(x, probability, yerr=probability_std, fmt='o')
axs[0,0].set_xlabel('Neighbor Number')
axs[0,0].set_ylabel('Probability')
axs[0,0].set_title('Probability with Standard Deviation')
axs[0,1].hist(search_zero_find_one[:,0])
axs[0,1].set_title('First Neighbor')
axs[0,2].hist(search_zero_find_one[:,1])
axs[0,2].set_title('Second Neighbor')
axs[1,0].hist(search_zero_find_one[:,2])
axs[1,0].set_title('Third Neighbor')
axs[1,1].hist(search_zero_find_one[:,3])
axs[1,1].set_title('Fourth Neighbor')
axs[1,2].hist(search_zero_find_one[:,4])
axs[1,2].set_title('Fifth Neighbor')
plt.show()


# Display the grid
fig, axs = plt.subplots(2, 2)
axs[0,0].imshow(data, cmap='gray', interpolation='none', origin='lower')
axs[0,0].set_title('Before Monte Carlo Simulation')
# print(type(data[100][129]))
transformed_coordinates, values = square_to_hex(data)
grid_size = data.shape[0]
a, b = 0, grid_size
markersize = 5

# Crop the image to remove the outer points
crop_x = []
crop_y = []
crop_values = []
for i in range(grid_size):
    for j in range(grid_size):
        if (transformed_coordinates[i,j,0] < grid_size//2) and (transformed_coordinates[i,j,0] > 0) and (transformed_coordinates[i,j,1] < grid_size//2 - 0.5) and (transformed_coordinates[i,j,1] > 0):
            crop_x.append(transformed_coordinates[i,j,0])
            crop_y.append(transformed_coordinates[i,j,1])
            crop_values.append(values[i,j])
            # print(values[i,j])
crop_x = np.array(crop_x)
crop_y = np.array(crop_y)
crop_values = np.array(crop_values)
# print(crop_x.shape, crop_y.shape)

# # Plot the image
# fig, axs = plt.subplots(2, 2, figsize=(10, 5))
# axs[0,0].imshow(data, cmap='gray')
# axs[0,0].set_title('Original Image')
# axs[0,1].scatter(transformed_coordinates[:, :, 0], np.flipud(transformed_coordinates[:, :, 1]), c=values, cmap='viridis', label='Transformed', marker='H', linewidth=markersize)
# axs[0,1].set_title('Transformed Image')
# axs[1,0].scatter(crop_x, crop_y, c = crop_values, cmap='viridis', label='Cropped', marker='H',)
# axs[1,1].hexbin(crop_x, crop_y, crop_values, gridsize=(grid_size//2-1, int(grid_size//2/np.sqrt(3))), cmap='viridis')
# axs[1,1].set_title('Cropped Hexagonal Grid')
# plt.show()

# print(crop_x.shape, crop_y.shape)
# select_point = 0
# fig, axs = plt.subplots(figsize=(6,6))
# axs.scatter(crop_x, crop_y, c = crop_values, cmap='viridis', label='Cropped', linewidths=0.2)
# axs.scatter(crop_x[select_point], crop_y[select_point], c = crop_values[select_point], cmap='viridis', label='Cropped', linewidths=3)
# select_point -= grid_size//2
# axs.scatter(crop_x[select_point], crop_y[select_point], c = crop_values[select_point], cmap='viridis', label='Cropped', linewidths=3)

# select_point -= grid_size//2
# axs.scatter(crop_x[select_point], crop_y[select_point], c = crop_values[select_point], cmap='viridis', label='Cropped', linewidths=3)

# axs.set_title('Cropped Hexagonal Grid')
# plt.show()








# FFT with interpolation
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

# # Generate a triangular lattice of points
# def generate_triangular_lattice(nx, ny, spacing):
#     x = np.arange(nx) * spacing
#     y = np.arange(ny) * spacing * np.sqrt(3) / 2
#     xv, yv = np.meshgrid(x, y)
#     xv[1::2] += spacing / 2  # Shift every other row
#     return xv.flatten(), yv.flatten()

# # Generate triangular lattice data points
# nx, ny = 50, 50  # Number of points in x and y directions
# spacing = 1.0
# x, y = generate_triangular_lattice(nx, ny, spacing)
# z = np.random.rand(len(x))  # Random values at lattice points

# # Define a regular grid for interpolation
# grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

# # Interpolate the data onto the regular grid
# grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# # Perform 2D FFT on the interpolated data
# fft_result = np.fft.fft2(grid_z)
# fft_result_shifted = np.fft.fftshift(fft_result)
# magnitude_spectrum = np.abs(fft_result_shifted)

# # Plot the original triangular lattice and its Fourier Transform
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.scatter(x, y, c=z, cmap='viridis')
# plt.title("Original Triangular Lattice Data")

# plt.subplot(1, 2, 2)
# plt.imshow(grid_z, cmap='viridis', origin='lower')
# plt.title("Fourier Transform")
# plt.show()
