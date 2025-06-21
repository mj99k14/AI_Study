import numpy as np


# np.random.seed(5)

# # x = np.random.randint(1,4,(2,2 ))
# # y = np.random.randint(1,4,(2,1))

# # print(f"{x}\n{y}\n{x + y}")
# x = np.array([[1,2],[3,4]])
# y = np.array([[2],[3]])

# print(x.shape)
# print(y.shape)

# print(f"{x} \n {y} \n { x @ y} ")


# # print(f"{x}\n{y}\n{x @ y}")

num_features = 4
num_samples = 1000

np.random.seed(5)

x = np.random.randn(num_samples,num_features) * 2
w_true = np.random.randint(1, 11,(num_features,1))
b_true = np.random.randn() * 0.5

y = x @ w_true + b_true

###############################################################
#초기의 w값은 달라야함
w = np.random.rand(num_features,1)
b = np.random.rand()
learn_rate = 0.01

gradient = np.zeros(num_features)
#예측 값 
for _ in range(1000): 
    predict_y = x @ w + b
    #오차 값
    error = predict_y - y
    #기울기
    gradient_w = x.T  @ error / num_samples
    gradient_b = error.mean()


    w = w - learn_rate * gradient_w
    b = b - learn_rate * gradient_b

print(f"{w_true},{b_true}")
print("---" *10)
print(f"{w},{b}")