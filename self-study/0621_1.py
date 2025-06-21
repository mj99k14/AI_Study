import matplotlib.pyplot as plt

# 백터 정의
a = [2,3]
b = [1,1]
ab=[a[0] + b[0], + a[1] +b[1]]

plt.quiver(0,0,a[0],a[1],angles = 'xy',scale_units = 'xy', scale =1, color ='red',lable='a = [2,3]')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='blue', label='b = [1,1]')
plt.quiver(0, 0, ab[0], ab[1], angles='xy', scale_units='xy', scale=1, color='green', label='a+b')

plt.xlim(0,5)
plt.xlim(0,5)
plt.grid()
plt.legend()
plt.title("백터 시각화")
plt.show()