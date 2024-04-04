import numpy as np

# Define the basis vectors of the hexagonal lattice
a1 = np.array([1, 0])  # x direction
a2 = np.array([1/2, np.sqrt(3)/2])  # 60 degrees from x direction

# Define the transformation function from hexagonal lattice to Cartesian coordinates
def hexagonal_to_cartesian(hx, hy):
    x = hx * a1[0] + hy * a2[0]
    y = hx * a1[1] + hy * a2[1]
    return x, y

# Define the inverse transformation function from Cartesian coordinates to hexagonal lattice
def cartesian_to_hexagonal(x, y):
    hx = (x * a2[1] - y * a2[0]) / (a1[0] * a2[1] - a1[1] * a2[0])
    hy = (y * a1[0] - x * a1[1]) / (a1[0] * a2[1] - a1[1] * a2[0])
    return hx, hy

# Example usage
hx, hy = -2, 2  # Example hexagonal lattice coordinates
x, y = hexagonal_to_cartesian(hx, hy)
print("Hexagonal lattice coordinates:", hx, hy)
print("Corresponding Cartesian coordinates:", x, y)

# Example of inverse transformation
x, y = 1.5, 1.5  # Example Cartesian coordinates
hx, hy = cartesian_to_hexagonal(x, y)
print("Cartesian coordinates:", x, y)
print("Corresponding hexagonal lattice coordinates:", hx, hy)
