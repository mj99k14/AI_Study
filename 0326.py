import numpy as np
import random
import matplotlib.pyplot as plt


# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]


#BGD

#1.H(X) = W * X 

#2.OPTIMIZER[ GD ]  W = w -slope of cost function for given w


#2.OPTIMIZER[ BGD ]  W = w -?
#2.OPTIMIZER[ SGD ]  W = w -?