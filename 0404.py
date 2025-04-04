import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error




np.random.seed(0)
x = np.random.rand(3, 1) * 10
y = 2.5 *  x + np.random.rand(3, 1) * 2
y = y.ravel()

# bar = np.random.rand(3,1)
# print(bar)
# print("-----" *10)
# print(bar.ravel())


model = SGDRegressor(max_iter = 1000,
                     learning_rate ='constant',
                     eta0 = 0.01,
                     penalty = None,
                     random_state = 0
                     )
#파라미터 => 학습을 해서 찾아가는 값(정답값) => 

#학습 실시
model.fit(x,y)


# import numpy as np

# bar = np.zeros((2))
# foo = np.zeros((3,2))
# pos = np.zeros((2,3,4))

# # print(f"bar.shape:{bar}")
# # print(f"foo.shape:{foo}")
# # print(f"pos.shape:{pos}")



# print(bar,"\n")

# print(foo,"\n")

# print(pos)
# np.set_printoptions(suppress=True,precision=2)

# x = np.random.rand(3, 1) * 10
# #h(x) = w *x +b
# pos = 2.5 * x 
# bar = np.random.randn(3, 1) * 2
# y = 2.5 * x + bar


# print(x)
# print("----" * 10)
# print(pos)
# print("----" * 10)
# print(bar)
# print("----" * 10)

# print(y)
# print("----" * 10)