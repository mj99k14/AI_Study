import numpy as np
import matplotlib.pyplot as plt

#data set

#input

#input data,features

#H(x)  -> input data :x1 ->xn
x_train = [ np.random.rand() * 10 for _ in range(50) ]
y_train = [val +  np.random.rand() * 5 for val in x_train]


for x,y in zip(x_train,y_train): #zip 묶어서 한 그룹으로 만들어서 2차원으로 만드어줌
    print(f"x:{x},y:{y}")


plt.scatter(x_train,y_train,color = "blue")
plt.show()