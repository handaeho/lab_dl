"""
P.210 ~ 213 식과 그림 참고

Batch Normalization(배치 정규화)
= 가중치의 초기값을 적절하게 설정하면 각 층의 활성화 값 분포가 적당히 퍼지며 학습이 원활히 수행된다.
따라서 각 층이 활성화를 적당히 퍼뜨릴수 있도록 '강제'하는 방법.

즉, 신경망의 각 층에 'mini-batch'를 전달 할 떄마다, '정규화'를 실행하도록 강제하는 방법이다.

input_data -> Affine -> [Batch_Norm] -> ReLU -> Affine -> [Batch_Norm] -> ReLU -> Affine -> Softmax -> output
~> 데이터 분포를 정규화 하는 '배치 정규화 계층'을 신경망에 삽입

'배치 정규화'는 학습 시, 'mini-batch'단위로 데이터 분포의 평균은 0, 분산은 1이 되도록 정규화한다.

장점
- 학습이 빠르다.
- 초기값에 크게 의존하지 않는다.
- 오버피팅을 억제한다.(드롭아웃 필요성 감소)

m개의 미니배치 입력 데이터에 대해 평균과 분산을 구하고, 평균=0, 분산=1이 되게 정규화 한다.
이때, 아주 작은 값인 '입실론(e)'을 분모에 더해 '0으로 나누기'를 방지한다.
u_B = 1/m * sigma_i(x_i)
d_B**2 = 1/m * sigma_i(x_i - u_B)**2
x_hat_i = (x_i - u_B) / sqrt(d_B**2 + e)

그리고 이 배치 정규화 계층마다 정규화 된 데이터에 고유한 확대화 이동 변환을 수행한다.
y <- r * x_hat_i + beta
r(gamma): 정규화된 미니배치를 확대 / 축소 (scale-up / scale-down)
beta: 정규화된 미니배치를 이동
r(gamma)=1, beta=0에서 시작해 학습하며 적합한 값으로 조정해 나간다.
"""
# p.213 그림 6-18을 그리세요.
# Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교
import matplotlib.pyplot as plt
import numpy as np

from ch06_Optimization.ex02_SGD import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend

from common.optimizer import Momentum, AdaGrad, Adam
from dataset.mnist import load_mnist

np.random.seed(110)

# 배치 정규화를 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size=784,
                                    hidden_size_list=[100, 100, 100, 100, 100],
                                    output_size=10,
                                    weight_init_std=0.3,
                                    use_batchnorm=True)
# 배치 정규화를 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100],
                                 output_size=10,
                                 weight_init_std=0.3,
                                 use_batchnorm=False)
# mini-batch iteration 회수 변경 -> 실험
# weight_init_std=0.01, 0.1, 0.5, 1.0 -> 실험

# 미니 배치를 20번 학습시키면서, 두 신경망에서 정확도(accuracy)를 기록
# -> 그래프

(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
# 학습 시간을 줄이기 위해서 학습 데이터의 개수를 줄임.
X_train = X_train[:1000]  # 데이터 1000개만 사용
Y_train = Y_train[:1000]

train_size = X_train.shape[0]
batch_size = 128
learning_rate = 0.01
iterations = 200

train_accuracies = []  # 배치 정규화를 사용하지 않는 신경망의 정확도를 기록
bn_train_accuracies = []  # 배치 정규화를 사용하는 신경망의 정확도를 기록

# optimizer = Sgd(learning_rate)
# 파라미터 최적화 알고리즘이 SGD가 아닌 경우에는 신경망 개수만큼 optimizer를 생성.
optimizer = Sgd(learning_rate)
bn_optimizer = Sgd(learning_rate)

# 학습하면서 정확도의 변화를 기록
for i in range(iterations):
    # 미니 배치를 랜덤하게 선택(0~999 숫자들 중 128개를 랜덤하게 선택)
    mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[mask]
    y_batch = Y_train[mask]

    # 배치 정규화를 사용하지 않는 신경망에서 gradient를 계산.
    gradients = neural_net.gradient(x_batch, y_batch)
    # 파라미터 업데이트(갱신) - W(가중치), b(편향)을 업데이트
    optimizer.update(neural_net.params, gradients)
    # 업데이트된 파라미터들을 사용해서 배치 데이터의 정확도 계산
    acc = neural_net.accuracy(x_batch, y_batch)
    # 정확도를 기록
    train_accuracies.append(acc)

    # 배치 정규화를 사용하는 신경망에서 같은 작업을 수행.
    bn_gradients = bn_neural_net.gradient(x_batch, y_batch)  # gradient 계산
    bn_optimizer.update(bn_neural_net.params, bn_gradients)  # W, b 업데이트
    bn_acc = bn_neural_net.accuracy(x_batch, y_batch)  # 정확도 계산
    bn_train_accuracies.append(bn_acc)  # 정확도 기록

    print(f'iteration #{i}: without={acc}, with={bn_acc}')

# 정확도 비교 그래프
x = np.arange(iterations)
plt.plot(x, train_accuracies, label='without BN')
plt.plot(x, bn_train_accuracies, label='using BN')
plt.legend()
plt.show()