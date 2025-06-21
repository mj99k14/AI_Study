from sklearn.model_selection import train_test_split
import numpy as np

#x : 입력 값 
x = np.random.rand(5 , 2) * 8 # rand 0이상 ~ 1미만, 5 * 2 열과행 만듬 ,* 계산을하면 그숫자 만큼 나옴(행열수 )범위

print(x)
#y : 출력값 (정답 값)
y = np.random.randint(0 , 4, size = 5) #0 이상 2 미만의 정수 중에서 무작위로 5개 뽑아서 배열
print(y)
#x_train, x_test, y_train, y_test = \
 #   train_test_split(x, y, test_size =0.2 , random_State = 1) #전체데이터 중에 20% 테스트 , 데이터를 나누는 무작위 시도 고정값 (동일한 값)



#for val in zip(x,y):
 