"""
과적합(overfitting)
= 모델이 학습 데이터는 정확하게 예측하지만, 학습되지 않은 데이터에 대해서 정확도가 떨어지는 현상.

overfitting이 나타나는 경우:
    1) 학습 데이터가 적은 경우
    2) 파라미터가 너무 많아서 표현력(representational power)이 너무 높은 모델

overfitting 되지 않도록 학습:
    1) regularization(정칙화, 규제): L1, L2-regularization
        손실(비용) 함수에 L1 규제(W) 또는 L2 규제(W**2)을 더해줘서, 파라미터(W, b)를 갱신(업데이트)할 때,
        파라미터가 더 큰 감소를 하도록 만드는 것.

        L2 규제 = L + (1/2) * lambda * ||W||**2 (lambda: 정규화의 세기를 조절하는 하이퍼 파라미터)
        가중치 W에 적용 -> W = W - lr * (dL/dW + lambda * W)
        ~~~> 가중치가 더 큰 값이 더 큰 감소를 일으킴.

        L1 규제 = L + lambda * ||W|| (lambda: 정규화의 세기를 조절하는 하이퍼 파라미터)
        가중치 W에 적용 -> W = W - lr * (dL/dW + lambda)
        ~~~> 모든 가중치가 일정한 크기로 감소됨.

    2) Dropout: 학습 중에 은닉층의 뉴런을 랜덤하게 골라서 삭제하고 학습시키는 방법.
    테스트를 할 때는 모든 뉴런을 사용함.

    overfitting을 줄이는 전략은 학습 시의 정확도를 일부러 줄이는 것임.
    -> 학습 데이터의 정확도와 테스트 데이터의 정확도 차이를 줄임.
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06_Optimization.ex02_SGD import Sgd
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist

np.random.seed(110)

# 데이터 준비
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# 신경망 생성
wd_rate = 0.
neural_net = MultiLayerNet(input_size=784,
                           hidden_size_list=[100, 100, 100, 100, 100],
                           output_size=10,
                           weight_decay_lambda=wd_rate)
# weight_decay_lambda: 가중치 감소에 사용할 상수 값

# 학습 데이터 개수를 300개로 제한 -> overfitting 만들기 위해서
X_train = X_train[:300]
Y_train = Y_train[:300]
X_test = X_test[:300]  # 실험 시간 줄이기 위해서
Y_test = Y_test[:300]

epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 경우
mini_batch_size = 100  # 1번 forward에 보낼 데이터 샘플 개수
train_size = X_train.shape[0]
iter_per_epoch = int(max(train_size / mini_batch_size, 1))
# 학습하면서 학습/테스트 데이터의 정확도를 각 에포크마다 기록
train_accuracies = []
test_accuracies = []

optimizer = Sgd(learning_rate=0.01)  # optimizer

for epoch in range(epochs):
    # indices = np.arange(train_size)
    # np.random.shuffle(indices)
    for i in range(iter_per_epoch):
        x_batch = X_train[(i * mini_batch_size):((i+1) * mini_batch_size)]
        y_batch = Y_train[(i * mini_batch_size):((i+1) * mini_batch_size)]
        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracies.append(train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    test_accuracies.append(test_acc)
    print(f'epoch #{epoch}: train={train_acc}, test={test_acc}')

x = np.arange(epochs)
plt.plot(x, train_accuracies, label='Train')
plt.plot(x, test_accuracies, label='Test')
plt.legend()
plt.title(f'Weight Decay (lambda={wd_rate})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()










