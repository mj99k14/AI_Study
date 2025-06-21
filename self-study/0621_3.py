# y = x^2 에서 경사하강법으로 최소값 찾기
x = 4.0  # 시작 위치
lr = 0.1  # 학습률

for i in range(10):
    grad = 2 * x  # 미분 값
    x = x - lr * grad  # 업데이트
    print(f"{i+1}회차: x = {x:.4f}, 기울기 = {grad:.4f}")
