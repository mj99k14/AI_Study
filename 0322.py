

#데이터
x = [1,4]
y = [3,6]

#초기가설
w = 0.0
b = 0.0

#학습률
lr = 0.1
#1. 예측률 구하기
y_pred = [w * xi +b for xi in x]

#2. 오차 구하기 예측값과 실제값 차이
error = [yi - ypi for yi,ypi in zip(y,y_pred)]



#for xi, yi in zip(x, y):
 #   y_pred = W * xi + b
  #  error = yi - y_pred


#3. cost 함수 구하기
cost = sum((e**2 for e in error))/len(x)  #e**2 제곱

#4. 기울기(w에 대한 미분값) 구하기
dw = -2 *sum(xi * ei for xi, ei in zip(x,error)) / len(x)

#5.절편에 (b에 대한 미분값) 구하기
db = -2 *sum(error)/len(x)
#6. w,b 업데이트 경사하강법 중심
w = w - lr * dw
b =b - lr * db


# 결과 출력
print("예측값:", y_pred)
print("오차:", error)
print("Cost:", cost)
print("기울기 dW:", dw)
print("절편 변화량 db:", db)
print("업데이트된 W:", w)
print("업데이트된 b:", b)