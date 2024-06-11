import numpy as np
import matplotlib.pyplot as plt
import random

# Create a 100x100 grid
grid_size = 256
total_elements = grid_size * grid_size

# Half ones and half zeros
part_elements = int(total_elements * 0.5)
grid = np.array([1] * part_elements + [0] * (total_elements - part_elements))

# Shuffle the grid to create a random distribution
np.random.shuffle(grid)

# Reshape into a 100x100 grid
grid = grid.reshape((grid_size, grid_size))
# grid[0:50,0:50] = 1

# Function to get nearest neighbors with periodic boundary conditions
def get_neighbors(grid, row, col):
    neighbors = []
    for r in range(row-1, row+2):
        for c in range(col-1, col+2):
            if (r == row and c == col) or (r==row-1 and c==col-1) or (r==row+1 and c==col+1):
                continue
            wrapped_r = r % grid_size
            wrapped_c = c % grid_size
            neighbors.append(grid[wrapped_r, wrapped_c])
    return neighbors

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


# Display the grid
fig, axs = plt.subplots(2, 2)
axs[0,0].imshow(grid, cmap='gray', interpolation='none')
axs[0,0].set_title('Before Monte Carlo Simulation')

for i in range(1000000):
    # Find a random "one" and "zero"
    ones_indices = np.argwhere(grid == 1)
    zeros_indices = np.argwhere(grid == 0)

    random_one = random.choice(ones_indices)
    random_zero = random.choice(zeros_indices)

    # Get neighbors for the random "one" and "zero"
    one_neighbors = get_neighbors(grid, random_one[0], random_one[1])
    zero_neighbors = get_neighbors(grid, random_zero[0], random_zero[1])

    one_count = one_neighbors.count(0)
    zero_count = zero_neighbors.count(1)

    e_before = one_count + zero_count
    e_after = 12 - e_before

    # Flip the random "one" and "zero" if energy decreases
    if e_after > e_before:    
        grid[random_one[0], random_one[1]] = 0
        grid[random_zero[0], random_zero[1]] = 1
    else:
        continue

# Save the grid data set after the for loop
np.savetxt('/Users/user1/Documents/GitHub/turing-patterns/grid_data_256_1e6_step.txt', grid)
print('Data after MC simluation is saved!')


    # # Display results
    # print(f"Random one at {random_one}: {grid[random_one[0], random_one[1]]}")
    # print(f"Neighbors of the random one: {one_neighbors}")
    # print(f"Number of zeros around the random one: {one_count}")

    # print(f"Random zero at {random_zero}: {grid[random_zero[0], random_zero[1]]}")
    # print(f"Neighbors of the random zero: {zero_neighbors}")
    # print(f"Number of ones around the random zero: {zero_count}")

# Display the grid
# axs[0,1].imshow(grid, cmap='gray', interpolation='none')
# axs[0,1].set_title('After Monte Carlo Simulation')

transformed_coordinates, values = square_to_hex(grid)
a, b = 0, grid_size
markersize = 5
# print(transformed_coordinates[:, :, 0].shape)
axs[0,1].scatter(transformed_coordinates[:, :, 0], np.flipud(transformed_coordinates[:, :, 1]), c=values, cmap='viridis', label='Transformed', marker='H', linewidth=markersize)
# hb = axs[1,1].hexbin(transformed_coordinates[:, :, 0], np.flipud(transformed_coordinates[:, :, 1]), gridsize=(int(grid_size*np.sqrt(3)/2), grid_size//2),mincnt=5, cmap='viridis')
# axs[1,1].set_title('Hexagonal Grid')
# plt.colorbar(hb)

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
print(crop_x.shape, crop_y.shape)
axs[1,0].scatter(crop_x, crop_y, c = crop_values, cmap='viridis', label='Cropped', marker='H',)
axs[1,1].hexbin(crop_x, crop_y, crop_values, gridsize=(grid_size//2-1, int(grid_size//2/np.sqrt(3))), cmap='viridis')
axs[1,1].set_title('Cropped Hexagonal Grid')
plt.show()

fig, axs = plt.subplots(figsize=(6,6))
axs.hexbin(crop_x, crop_y, crop_values, gridsize=(grid_size//2-1, int(grid_size//2/np.sqrt(3))), cmap='viridis')
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.spatial

# def generate_hex_grid(x_range, y_range, spacing):
#     dx = spacing * 3/2
#     dy = spacing * np.sqrt(3)
    
#     x_min, x_max = x_range
#     y_min, y_max = y_range
    
#     grid_x = []
#     grid_y = []
    
#     for x in np.arange(x_min, x_max, dx):
#         for y in np.arange(y_min, y_max, dy):
#             grid_x.append(x)
#             grid_y.append(y)
#             grid_x.append(x + spacing * 3/2)
#             grid_y.append(y + spacing * np.sqrt(3)/2)
    
#     return np.array(grid_x), np.array(grid_y)

# # Example dataset
# x = np.random.rand(100) * 10  # replace with your x-coordinates
# y = np.random.rand(100) * 10  # replace with your y-coordinates

# # Define the range and spacing for the hexagonal grid
# x_range = (x.min(), x.max())
# # print(len(x))
# y_range = (y.min(), y.max())
# spacing = 1.0  # Adjust the spacing as needed

# # Generate the hexagonal grid
# grid_x, grid_y = generate_hex_grid(x_range, y_range, spacing)

# # Plotting to visualize
# plt.figure(figsize=(6, 6))
# plt.hexbin(grid_x, grid_y, gridsize=int((x.max()-x.min())/spacing), cmap='viridis')
# plt.scatter(x, y, color='red', s=1)  # Plot the original scatter data
# plt.show()


