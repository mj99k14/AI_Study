import numpy as np
import matplotlib.pyplot as plt

x =np.linspace(-5,5,100)
y =x**2

plt.plot(x,y)
plt.scatter(2,4,color='red')
plt.grid()
plt.title("y= x2 그래프")
plt.show()