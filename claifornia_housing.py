import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 데이터 로딩
# fetch_california_housing() 함수로 입력 데이터(X)와 타겟값(y)을 불러옴
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. 학습/테스트용 데이터 분할 (훈련 : 80%, 테스트 : 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 입력 특성 정규화 (표준화 : 평균 0, 표준편차 1로 맞춤)
# SGD는 입력값 크기에 민감하므로 반드시 정규화가 필요
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. SGDRegressor 모델 정의 및 학습 / 변경 완 (학습률(eta0)과 학습률 스케줄(learning_rate)을 바꾸고, 정규화(penalty)와 수렴 조건(tol)을 조정해서 모델이 더 잘 학습되고 과적합도 막도록 바꾼 거야.)
model = SGDRegressor(
    max_iter=1000, #EPOCH수
    tol=1e-4,  # 더 정밀한 수렴 기준
    eta0=0.01,  # 약간 더 높은 학습률
    learning_rate='invscaling',  # 학습률 점점 감소
    penalty='l2',  # 기본 L2 정규화
    random_state=42 #랜덤 시드 고정(결과 재현 가능하게 함 )
)

model.fit(X_train_scaled, y_train)

# 5. 예측 및 평가
y_pred = model.predict(X_test_scaled)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 score: {r2_score(y_test, y_pred):.4f}")

# 6. 회귀 계수 출력 (각 특성이 결과에 얼마나 영향을 주는지 보여줌)
print("\n휘귀 계수 (weights):")
for name, coef in zip(feature_names, model.coef_):
    # 각 특성 이름과 회귀 계수를 정렬된 형태로 출력
    print(f"{name:<20}: {coef:>20,.2f}")
    
# 절편(bias term) 출력
print(f"절편 (bias): {model.intercept_[0]:,.2f}")

# 7. 예측 결과 시각화 (실제값 VS 예측값 산점도)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Value")
plt.title("SGDRegressor Prediction vs Actual")
plt.grid(True)
plt.show()