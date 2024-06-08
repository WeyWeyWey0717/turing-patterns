import numpy as np
import matplotlib.pyplot as plt

first_x, first_y, first_value = [], [], []
second_x, second_y, second_value = [], [], []
third_x, third_y, third_value = [], [], []
fourth_x, fourth_y, fourth_value = [], [], []
fifth_x, fifth_y, fifth_value = [], [], []
# sixth_x, sixth_y, sixth_value = [], [], []


for i in range(-3,4):
    for j in range(-3,4):
        if abs((j+1/2*i)**2 + (i*np.sqrt(3)/2)**2 - 1) < 0.01:
            first_x.append(i)
            first_y.append(j)
            first_value.append(1)
        if abs((j+1/2*i)**2 + (i*np.sqrt(3)/2)**2 - 3) < 0.01:
            second_x.append(i)
            second_y.append(j)
            second_value.append(3)
        if abs((j+1/2*i)**2 + (i*np.sqrt(3)/2)**2 - 4) < 0.01:
            third_x.append(i)
            third_y.append(j)
            third_value.append(4)
        if abs((j+1/2*i)**2 + (i*np.sqrt(3)/2)**2 - 7) < 0.01:
            fourth_x.append(i)
            fourth_y.append(j)
            fourth_value.append(7)
        if abs((j+1/2*i)**2 + (i*np.sqrt(3)/2)**2 - 9) < 0.01:
            fifth_x.append(i)
            fifth_y.append(j)
            fifth_value.append(9)


first_x = np.array(first_x)
first_y = np.array(first_y)
first_value = np.array(first_value)
second_x = np.array(second_x)
second_y = np.array(second_y)
second_value = np.array(second_value)
third_x = np.array(third_x)
third_y = np.array(third_y)
third_value = np.array(third_value)
fourth_x = np.array(fourth_x)
fourth_y = np.array(fourth_y)
fourth_value = np.array(fourth_value)
fifth_x = np.array(fifth_x)
fifth_y = np.array(fifth_y)
fifth_value = np.array(fifth_value)
print(fifth_x)
fig, axs = plt.subplots(figsize=(8,8))
trans_to_hex = [1/2, np.sqrt(3)/2] # or [0 ,1] if no transformation
axs.scatter(first_x +trans_to_hex[0]*first_y,  trans_to_hex[1]*first_y, first_value, marker='o')
axs.scatter(second_x+trans_to_hex[0]*second_y, trans_to_hex[1]*second_y, second_value, marker='H')
axs.scatter(third_x +trans_to_hex[0]*third_y,  trans_to_hex[1]*third_y, third_value, marker='s')
axs.scatter(fourth_x+trans_to_hex[0]*fourth_y, trans_to_hex[1]*fourth_y, fourth_value, marker='x')
axs.scatter(fifth_x +trans_to_hex[0]*fifth_y,  trans_to_hex[1]*fifth_y, fifth_value, marker='D')

plt.show()