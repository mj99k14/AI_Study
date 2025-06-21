import numpy as np

# #h(x) = wx1 +wx2 +wx3 +b

# num_features = 3
# num_sample =2
# x = np.random.rand(num_features,num_sample)

# np.random.seed(1)
# np.set_printoptions(suppress=False,precision=3)
# # print(f"{kin.shape}{bar.shape}{foo.shape}")

# w_true = np.random.randint(1,10,num_features)
# b_true = np.random.randn() * 0.5

# y = x[:,0] * w_true[0] + x[:,1] * w_true[1] + x[:,2] * w_true[2] + b_true
# print(f"[{w_true},{b_true},{y}")

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(x)
print()
print(x.T)