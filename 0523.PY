import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


bar = np.array([val for val in range(1, 21)])

print(bar.reshape(-1, 4))


#Prediction

#z = w * x +b

#sigmoid(z) ->1 /(1+e^-z)


#error

# Get gradient for w and b

#update parameter values of each w and b

#display the loss value of the current epoch