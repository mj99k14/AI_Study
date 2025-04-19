import numpy as np

#1. 데이터 생성
np.random.seed(42)
num_Sample = 10 #데이터의수
num_features = 2#특징의 수
x =np.random.rand(num_Sample, num_features) * 10
true_w =np.array([3,5])
true_b = 10
noise = np.random.randn(num_Sample) * 0.5
y = x @ true_w + true_b + noise # 현실세계데이터랑 어긋날수있기때문에


#파라미터 초기화
w = np.random.rand(num_features)
b = np.random.rand()
lr = 0.03
epochs = 10000

#3.학습 루프
for epoch in range(epochs):
    pred = x @ w + b 
    error =  pred -y 
    
    grad_w = x.T @ error /num_Sample
    grad_b = error.mean()  #평균값 


    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 50 == 0:
        loss = (error **2).mean()
        print(f"[Epoch {epoch}] 평균 손실 :{loss:.4f}")


print("\n학습된 가중치",w)
print("학습된 편향")


    