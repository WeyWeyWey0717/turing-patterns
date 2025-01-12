import numpy as np
import matplotlib.pyplot as plt
import time

grid_size = 64
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)

interval = [1.5, 1.75, 2]

X_hex = X + 0.5 * Y
Y_hex = (np.sqrt(3)/2) * Y
alpha = 1 # interval[0]
Y_hex = alpha * Y_hex

# Number of points
num_points = X.size  # 256*256 = 65536

# Create an array with half zeros and half ones
values = np.zeros(num_points, dtype=int)
values[:num_points//4] = 1

# Shuffle the values to randomize their positions
np.random.shuffle(values)

# Optionally, reshape to 256x256 if you want a 2D array
values_2d = values.reshape(X.shape)

# Now, values_2d[i,j] corresponds to the point at X[i,j], Y[i,j]

# Plot the points colored by their value
X_flat = X_hex.flatten()
Y_flat = Y_hex.flatten()
print(X_flat)
print(X_flat.max())
# Color map: points with value "0" will be one color, "1" another
# color_map = ['red' if v == 0 else 'blue' for v in values]

'''# Randomly choose one site with value 1
indices_with_1 = np.where(values == 1)[0]
chosen_index = np.random.choice(indices_with_1)

# Calculate the coordinates of the chosen site
chosen_x_1 = X_flat[chosen_index]
chosen_y_1 = Y_flat[chosen_index]'''


# Randomly choose one site with value 1 that is not at the edge
indices_with_1 = np.where(values == 1)[0]
indices_with_0 = np.where(values == 0)[0]

# Define a function to check if a point is at the edge
def is_not_at_edge(index, X_flat, Y_flat, margin=1):
    x, y = X_flat[index], Y_flat[index]
    return (x > margin) and (x < X_flat.max() - margin) and (y > margin) and (y < Y_flat.max() - margin)

# Filter indices to exclude edge points
filtered_indices_1 = [index for index in indices_with_1 if is_not_at_edge(index, X_flat, Y_flat)]
filtered_indices_0 = [index for index in indices_with_0 if is_not_at_edge(index, X_flat, Y_flat)]


steps = int(1e7)
time_start = time.time()
switch_index = []
for step in range(steps):
    # Randomly choose one site from the filtered indices
    chosen_index_1 = np.random.choice(filtered_indices_1)
    chosen_index_0 = np.random.choice(filtered_indices_0)

    # Calculate the coordinates of the chosen site
    chosen_x_1 = X_flat[chosen_index_1]
    chosen_y_1 = Y_flat[chosen_index_1]
    chosen_x_0 = X_flat[chosen_index_0]
    chosen_y_0 = Y_flat[chosen_index_0]

    # Calculate the coordinates of the 6 nearest neighbors in a hexagonal grid
    neighbors_1 = [
        (chosen_x_1 + 1, chosen_y_1),
        (chosen_x_1 - 1, chosen_y_1),
        (chosen_x_1 + 0.5, chosen_y_1 + np.sqrt(3)/2 * alpha),
        (chosen_x_1 - 0.5, chosen_y_1 + np.sqrt(3)/2 * alpha),
        (chosen_x_1 + 0.5, chosen_y_1 - np.sqrt(3)/2 * alpha),
        (chosen_x_1 - 0.5, chosen_y_1 - np.sqrt(3)/2 * alpha)
    ]

    neighbors_0 = [
        (chosen_x_0 + 1, chosen_y_0),
        (chosen_x_0 - 1, chosen_y_0),
        (chosen_x_0 + 0.5, chosen_y_0 + np.sqrt(3)/2 * alpha),
        (chosen_x_0 - 0.5, chosen_y_0 + np.sqrt(3)/2 * alpha),
        (chosen_x_0 + 0.5, chosen_y_0 - np.sqrt(3)/2 * alpha),
        (chosen_x_0 - 0.5, chosen_y_0 - np.sqrt(3)/2 * alpha)
    ]

    # Print the coordinates of the chosen site and its neighbors
    # print("")
    # print("Chosen site coordinates:")
    # print("1: ",[chosen_x_1, chosen_y_1], "0: ",[chosen_x_0, chosen_y_0])
    # print("Coordinates of the 6 nearest neighbors:")
    neighbor_1_values = []
    neighbor_1_distance_values = []
    for neighbor in neighbors_1:
        # Find the values of the 6 nearest neighbors
        # Find the closest grid point to the neighbor coordinates
        distances = np.sqrt((X_flat - neighbor[0])**2 + (Y_flat - neighbor[1])**2) # NOT CONSIDER BOUNDARY YET
        neighbor_1_distance_values.append(np.sqrt((chosen_x_1 - neighbor[0])**2 + (chosen_y_1 - neighbor[1])**2))
        closest_index = np.argmin(distances)
        neighbor_1_values.append(values[closest_index])
        # print(neighbor, neighbor_1_values[-1])
    # Print the values of the 6 nearest neighbors
    # print("Values of the 6 nearest neighbors of 1:")
    # print(neighbor_1_values)
    # print(neighbor_1_distance_values)

    neighbor_0_values = []
    neighbor_0_distance_values = []
    for neighbor in neighbors_0:
        # Find the values of the 6 nearest neighbors
        # Find the closest grid point to the neighbor coordinates
        distances = np.sqrt((X_flat - neighbor[0])**2 + (Y_flat - neighbor[1])**2) # NOT CONSIDER BOUNDARY YET
        neighbor_0_distance_values.append(np.sqrt((chosen_x_0 - neighbor[0])**2 + (chosen_y_0 - neighbor[1])**2))
        closest_index = np.argmin(distances)
        neighbor_0_values.append(values[closest_index])
        # print(neighbor, neighbor_0_values[-1])
    # Print the values of the 6 nearest neighbors
    # print("Values of the 6 nearest neighbors of 0:")
    # print(neighbor_0_values)
    # print(neighbor_0_distance_values)

    E_before = 0
    E_after = 0
    for i in range(len(neighbor_1_values)):
        if neighbor_1_values[i] == 0:
            E_before += -1/neighbor_1_distance_values[i]
            E_after += 1/neighbor_1_distance_values[i]
        if neighbor_1_values[i] == 1:
            E_before += 1/neighbor_1_distance_values[i]
            E_after += -1/neighbor_1_distance_values[i]

    for i in range(len(neighbor_0_values)):
        if neighbor_0_values[i] == 0:
            E_before += 1/neighbor_0_distance_values[i]
            E_after += -1/neighbor_0_distance_values[i]
        if neighbor_0_values[i] == 1:
            E_before += -1/neighbor_0_distance_values[i]
            E_after += 1/neighbor_0_distance_values[i]
    # print(E_before, E_after)

    if E_after < E_before:
        values[chosen_index_1] = 0
        values[chosen_index_0] = 1
        switch_index.append(1)
    else:
        switch_index.append(0)
        # print("Switched")
    if step % 1000000 == 0:
        print(step)

color_map = ['red' if v == 0 else 'blue' for v in values]

end_time = time.time()
print("Time elapsed: ", end_time - time_start)

plt.plot(switch_index)
plt.show()

np.savetxt(f'/Users/user1/Documents/GitHub/turing-patterns/20241211_grid_data_{grid_size}_{steps}_step_c=0_25_alpha=_{alpha}.txt', values)
print('Data after MC simluation is saved!')

plt.figure(figsize=(6,6))
markersize = 3
plt.scatter(X_flat, Y_flat, c=color_map, marker='H', linewidth=markersize)
plt.title('Coordinates with Assigned Values (0 or 1)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



