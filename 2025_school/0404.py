import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
#1. 데이터 생성 파트
np.random.seed(0) #랜덤 결과를 고정시킴
x = np.random.rand(3, 1) * 10
y = 2.5 *  x + np.random.rand(3, 1) * 2 #기울기 *2를 해주는 이유는 노이즈를 주기위해
y = y.ravel() #1차원 배열로 만들어줌

# bar = np.random.rand(3,1)
# print(bar)
# print("-----" *10)
# print(bar.ravel())

#2. 모델 학습 파트 
model = SGDRegressor(
                     max_iter = 1000, #최대학습을 1000번을 반복 
                     learning_rate ='constant', #고정된 값 유지
                     eta0 = 0.01, #실제 학습률
                     penalty = None, #규제를 안주겠다
                     random_state = 0 #재현성 유지지
                     )
#파라미터 => 학습을 해서 찾아가는 값(정답값) => 

#학습 실시
model.fit(x,y) #경사하강법 으로 학습시작작
#평가
#loss 값 확인 / cost function확인

# 3. 평가 파트 (잘 배웠는지 확인!)
y_pred = model.predict(x) #x값 예측
mse = mean_squared_error(y, y_pred) 


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