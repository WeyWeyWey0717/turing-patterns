import numpy as np
import matplotlib.pyplot as plt


# Define values for each point in the grid
values = np.fliplr(np.loadtxt('test1.txt', dtype=float))

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

# Plot original grid with values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Grid with Values')
plt.scatter(xx, yy, c=np.flip(values), cmap='plasma', label='Original', marker='o')
# plt.imshow(np.fliplr(values), cmap='plasma', label='Original')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Value')
plt.grid(True)
plt.legend()

# Plot transformed grid with values
plt.subplot(1, 2, 2)
plt.title('Transformed Hexagonal Grid with Values')
plt.scatter(transformed_coordinates[:, :, 0], transformed_coordinates[:, :, 1], c=np.flip(values), cmap='plasma', label='Transformed', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
