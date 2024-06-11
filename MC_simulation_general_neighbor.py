import numpy as np
import matplotlib.pyplot as plt
import random

# Create a 100x100 grid
grid_size = 100
total_elements = grid_size * grid_size

# Half ones and half zeros
half_elements = total_elements // 2
grid = np.array([1] * half_elements + [0] * half_elements)

# Shuffle the grid to create a random distribution
np.random.shuffle(grid)

# Reshape into a 100x100 grid
grid = grid.reshape((grid_size, grid_size))
# grid[0:50,0:50] = 1

neighbor_distance = [1, 3, 4, 7, 9] # in square unit

def LJ_potential_energy(r):
    return (1/r**12 - 2/r**6)

# Function to get nearest neighbors with periodic boundary conditions
def get_neighbors(grid, row, col):
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




for i in range(1):
    # Find a random "one" and "zero"
    ones_indices = np.argwhere(grid == 1)
    zeros_indices = np.argwhere(grid == 0)

    random_one = random.choice(ones_indices)
    random_zero = random.choice(zeros_indices)

    # Get neighbors for the random "one" and "zero"
    one_neighbors, one_neighbors_coordinates = get_neighbors(grid, random_one[0], random_one[1])
    zero_neighbors, zero_neighbors_coordinates = get_neighbors(grid, random_zero[0], random_zero[1])
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

    print(one_neighbor_one, one_neighbor_zero)

    # # Calculate energy before and after flipping
    e_before, e_after = 0, 0
    for i in range(len(one_neighbor_zero)):
        e_before += 0.5*LJ_potential_energy(np.sqrt(neighbor_distance[i]))*(one_neighbor_one[i] - one_neighbor_zero[i] + zero_neighbor_zero[i] - zero_neighbor_one[i])
        e_after += 0.5*LJ_potential_energy(np.sqrt(neighbor_distance[i]))*(one_neighbor_zero[i] - one_neighbor_one[i] + zero_neighbor_one[i] - zero_neighbor_zero[i])

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



transformed_coordinates, values = square_to_hex(grid)
a, b = 0, grid_size
markersize = 5
# print(transformed_coordinates[:, :, 0].shape)
# hb = axs[1,1].hexbin(transformed_coordinates[:, :, 0], np.flipud(transformed_coordinates[:, :, 1]), gridsize=(int(grid_size*np.sqrt(3)/2), grid_size//2),mincnt=5, cmap='viridis')
# axs[1,1].set_title('Hexagonal Grid')
# plt.colorbar(hb)

crop_x = []
crop_y = []
crop_values = []
for i in range(grid_size):
    for j in range(grid_size):
        if (transformed_coordinates[i,j,0] < grid_size) and (transformed_coordinates[i,j,0] > grid_size//2) and (transformed_coordinates[i,j,1] < grid_size//2 - 0.5) and (transformed_coordinates[i,j,1] > 0):
            crop_x.append(transformed_coordinates[i,j,0])
            crop_y.append(transformed_coordinates[i,j,1])
            crop_values.append(values[i,j])
            # print(values[i,j])
crop_x = np.array(crop_x)
crop_y = np.array(crop_y)
crop_values = np.array(crop_values)
# print(crop_x.shape, crop_y.shape)


# # Display the grid
# fig, axs = plt.subplots(2, 2)
# axs[0,0].imshow(grid, cmap='gray', interpolation='none', origin='lower')
# axs[0,0].set_title('Before Monte Carlo Simulation')
# axs[0,1].imshow(grid, cmap='gray', interpolation='none', origin='lower')
# axs[0,1].set_title('After Monte Carlo Simulation')
# axs[1,0].scatter(transformed_coordinates[:, :, 0], (transformed_coordinates[:, :, 1]), c=values, cmap='viridis', label='Transformed', marker='H', linewidth=markersize)
# axs[1,1].scatter(crop_x, crop_y, c = crop_values, cmap='viridis', label='Cropped', marker='H',)
# axs[1,1].set_title('Cropped Hexagonal Grid')
# plt.show()

# fig, axs = plt.subplots(figsize=(6,6))
# axs.hexbin(crop_x, crop_y, crop_values, gridsize=(grid_size//2-1, int(grid_size//2/np.sqrt(3))), cmap='viridis')
# plt.show()



# # import numpy as np
# # import matplotlib.pyplot as plt
# # import scipy.spatial

# # def generate_hex_grid(x_range, y_range, spacing):
# #     dx = spacing * 3/2
# #     dy = spacing * np.sqrt(3)
    
# #     x_min, x_max = x_range
# #     y_min, y_max = y_range
    
# #     grid_x = []
# #     grid_y = []
    
# #     for x in np.arange(x_min, x_max, dx):
# #         for y in np.arange(y_min, y_max, dy):
# #             grid_x.append(x)
# #             grid_y.append(y)
# #             grid_x.append(x + spacing * 3/2)
# #             grid_y.append(y + spacing * np.sqrt(3)/2)
    
# #     return np.array(grid_x), np.array(grid_y)

# # # Example dataset
# # x = np.random.rand(100) * 10  # replace with your x-coordinates
# # y = np.random.rand(100) * 10  # replace with your y-coordinates

# # # Define the range and spacing for the hexagonal grid
# # x_range = (x.min(), x.max())
# # # print(len(x))
# # y_range = (y.min(), y.max())
# # spacing = 1.0  # Adjust the spacing as needed

# # # Generate the hexagonal grid
# # grid_x, grid_y = generate_hex_grid(x_range, y_range, spacing)

# # # Plotting to visualize
# # plt.figure(figsize=(6, 6))
# # plt.hexbin(grid_x, grid_y, gridsize=int((x.max()-x.min())/spacing), cmap='viridis')
# # plt.scatter(x, y, color='red', s=1)  # Plot the original scatter data
# # plt.show()









# A simple searching nearest neighbors in 2D array
# # import numpy as np

# # # Example data points
# # data_points = np.array([
# #     [1, 2],
# #     [2, 3],
# #     [3, 4],
# #     [4, 5],
# #     [5, 6],
# #     [7, 8],
# #     [8, 9],
# #     [10, 10]
# # ])

# # # Selected data point
# # selected_point = np.array([3, 3])

# # # Calculate the Euclidean distance between the selected point and all other points
# # distances = np.linalg.norm(data_points - selected_point, axis=1)

# # # Get the indices of the sorted distances
# # sorted_indices = np.argsort(distances)

# # # Select the 6 nearest neighbors (excluding the selected point itself if present in the dataset)
# # nearest_neighbors_indices = sorted_indices[0:6]

# # # Get the nearest neighbors
# # nearest_neighbors = data_points[nearest_neighbors_indices]

# # print("Nearest Neighbors:")
# # print(nearest_neighbors)




import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def extend_points(data_points, x_box_size, y_box_size):
    # Duplicate the data points in all periodic directions
    extended_points = []
    x_shifts = [-x_box_size, 0, x_box_size]
    y_shifts = [-y_box_size, 0, y_box_size]
    for dx in x_shifts:
        for dy in y_shifts:
            shift = np.array([dx, dy])
            extended_points.append(data_points + shift)
    return np.vstack(extended_points)

# Example 100x100 grid of data points
crop_x = crop_x - crop_x[0]
x_box_size = 50 #max(crop_x) - min(crop_x) + (crop_x[0])/2
y_box_size = 50 #max(crop_y) - min(crop_y)# - crop_y[0]/2
# data_points = np.random.rand(10503, 2) * box_size  # Generating random 100x100 points in a unit box

data_points = np.column_stack((crop_x-crop_x[0], crop_y))
print(crop_x[0])
# print(data_points.shape)
# print(crop_x.shape, crop_y.shape)

# Selected data point
# selected_point = np.array([75, 25])
selected_point = np.array([crop_x[0], crop_y[0]])
# selected_point = random.choice(data_points)

# Extend the data points to account for periodic boundary conditions
extended_points = extend_points(data_points, x_box_size, y_box_size)



# Create a KDTree for the extended data points
tree = KDTree(extended_points)

# Query the KDTree for the k nearest neighbors
distances, indices = tree.query(selected_point, k=7)

# Check if the selected point itself is in the result and remove it
nearest_neighbors_indices = indices[indices != tree.query_ball_point(selected_point, 1e-3)[0]][:6]  # Using a small radius to find the exact match

# Get the nearest neighbors (original indices)
nearest_neighbors = extended_points[nearest_neighbors_indices]

# Map neighbors back to the original space
nearest_neighbors = nearest_neighbors % x_box_size

plt.figure(figsize=(6, 6))
# plt.scatter(extended_points[:, 0], extended_points[:, 1], color='blue', s=1)
plt.scatter(data_points[:, 0], data_points[:, 1], color='blue', s=1)
plt.scatter(selected_point[0], selected_point[1], color='red', s=10)
plt.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color='green', s=5)
plt.show()

# print("Nearest Neighbors:")
# print(nearest_neighbors)
