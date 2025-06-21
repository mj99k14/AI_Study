import numpy as np

num_of_samples = 5
num_of_features = 2

np.random.seed(1)
np.set_printoptions(False,suppress=True)
x = np.random.rand(num_of_samples, num_of_features)  * 10
x_ture = [5, 3]
b_ture = 4

print(x)
print("----------------------")
print(x[:, 0])
print("----------------------")
print(x[:, 1])