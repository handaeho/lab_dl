"""
MNIST 데이터를 사용한 가중치 초기값에 대한 신경망 성능 비교
"""
import numpy as np
import matplotlib.pyplot as plt


from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Adam
from common.util import smooth_curve
from dataset.mnist import load_mnist

# 테스트 조건 세팅
weight_init_types = {
        'std=0.01': 0.01, # 가중치 초기값 N(0, 0.01)
        'Xavier': 'sigmoid', # 가중치 초기값 N(0, sqrt(1/n))
        'He': 'relu' # 가중치 초기값 N(0, sqrt(2/n))
    }

# 각 조건 별로 테스트할 신경망 생성
neural_nets = dict()
train_losses = dict()
for key, weight_type in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10,
                                     weight_init_std=weight_type)
    train_losses[key] =[] # 테스트하며 계산되는 loss 계산 후 저장을 위한 빈 리스트 생성

# MNIST 데이터 load
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

iterations = 2000 # 총 학습 횟수
batch_size = 128 # 1번 학습에 사용할 샘플 개수(mini_batch)

# optimizer를 변경하며 테스트
#optimizer = SGD(learning_rate=0.01) # 파라미터(W/b) 최적화 알고리즘
optimizer = Adam() # 파라미터(W/b) 최적화 알고리즘

train_size = X_train.shape[0] # 60,000개의 train_set

for i in range(iterations): # 2000번 반복하며
    np.random.seed(109)
    # train_data(X_train)과 train_label(Y_train)에서 미니 배치 사이즈만큼 랜덤하게 데이터 선별
    batch_mask = np.random.choice(train_size, batch_size)
    # ~> 0 ~ 59,999 사이의 숫자(train_size)들에서 128개(batch_size)씩 랜덤하게 선택

    # 학습에 사용할 미니 배치 데이터와 레이블을 선택
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]

    # 신경망 종류마다 테스트 반복
    for key, net in neural_nets.items():
        # gradient 계산
        gradients = net.gradient(X_batch, Y_batch)
        # 파라미터(W/b) 업데이트
        optimizer.update(net.params, gradients)
        # loss 계산하고, train_losses 리스트에 추가
        loss = net.loss(X_batch, Y_batch)
        train_losses[key].append(loss)
    # loss 출력(100번 학습마다)
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key, loss_list in train_losses.items():
            print(key, ":", loss_list[-1])
            # 표준편차 'std=0.01'인 초기값 에서는 loss가 거의 감소하지 않았지만,
            # 표준편차가 'Xavier'과'He'인 초기값에서는 loss가 크게 감소함을 볼 수 있다.

# 반복횟수에 따른 loss 변화 시각화(x축 반복횟수, y축 loss)
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_losses[key]), marker=markers[key], label=key)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title('Iterations - Loss Weight Compare')
plt.legend()
plt.show()
