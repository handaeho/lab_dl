"""
MNIST 데이터에 지금까지 구현한 파라미터 최적화 알고리즘 6개를 적용해 성능을 비교해 보자.
~> 지표? 손실(loss), 정확도(accuracy)

모델은 'ch05_Back_Propagation.ex10_MNIST_Two_Layer_NN_Propagation'에서의 'TwoLayerNetwork' 클래스
"""
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from ch05_Back_Propagation.ex10_MNIST_Two_Layer_NN_Propagation import TwoLayerNetwork
from ch06_Optimization.ex02_SGD import Sgd
from ch06_Optimization.ex03_Momentum import Momentum
from ch06_Optimization.ex04_AdaGrad import AdaGrad
from ch06_Optimization.ex05_Adam import Adam
from ch06_Optimization.ex06_rmsprop import RMSProp
from ch06_Optimization.ex07_nesterov import Nesterov


if __name__ == '__main__':
    # MNIST 데이터 load
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 최적화 알고리즘을 구현한 클래스의 객체(인스턴스)를 dict에 저장
    optimizers = dict()
    optimizers['SGD'] = Sgd()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSProp'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()

    # hidden_layer 1개, output_layer 1개로 이루어진 신경망을 optimizer 개수만큼 생성
    neural_nets = dict()
    train_losses = dict() # 각 optimizer별로 train_set의 손실량을 계산해 저장할 dict
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
        train_losses[key] = [] # 각 optimizer 별로 loss값의 변화 history를 저장할 리스트

    # 각각의 신경망을 학습시키며 loss 기록
    np.random.seed(108)
    iterations = 2000 # 학습 총 횟수
    batch_size = 128 # 한번의 학습에서 사용할 미니 배치 사이즈
    train_size = X_train.shape[0] # 60,000개의 train_set

    for i in range(iterations): # 총 2000번 학습 반복
        # train_data(X_train)과 train_label(Y_train)에서 미니 배치 사이즈만큼 랜덤하게 데이터 선별
        batch_mask = np.random.choice(train_size, batch_size)
        # ~> 0 ~ 59,999 사이의 숫자(train_size)들에서 128개(batch_size)씩 랜덤하게 선택

        # 학습에 사용할 미니 배치 데이터와 레이블을 선택
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]

        # 선택된 학습 데이터와 레이블을 사용해서 gradient들을 계산
        for key in optimizers:
            # 각각의 optimizer를 사용해 gradient 계산
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # 계산한 gradient로 각각의 optimizer의 파라미터 업데이트
            # ~> 신경망이 가지고 있는 파라미터(Weight/bias)와 계산한 gradient를 넘겨줌
            optimizers[key].update(neural_nets[key].params, gradients)
            # 각 optimizer에 적용된 미니 배치에 대한 loss 계산
            loss = neural_nets[key].loss(X_batch, Y_batch)
            train_losses[key].append(loss) # 계산한 각 loss를 dict 타입인 train_losses에 저장

        # 학습을 100번 반복할 때마다 계산된 loss 확인
        if i % 100 == 0:
            print(f'========== Training #{i} ==========')
            for key in optimizers:
                print(key, ':', train_losses[key][-1])
                # ~> train_losses dict에서 key로 꺼내 리스트를 만들고, 그 마지막 원소를 꺼냄.
                # ========== Training #0 ==========
                # SGD : 2.3031851652469175
                # Momentum : 2.3031851652469175
                # AdaGrad : 2.275731197034728
                # Adam : 2.2757110755217176
                # RMSProp : 2.328150906907833
                # Nesterov : 2.303055316035674
                # ...
                # ========== Training #1900 ==========
                # SGD : 0.4856691636648474
                # Momentum : 0.19434487727704125
                # AdaGrad : 0.204040592687178
                # Adam : 0.0644752923498066
                # RMSProp : 0.015570117313334686
                # Nesterov : 0.18202973492373098

    # 그럼 optimizer별로 학습횟수에 따른 loss의 변화를 그래프로 시각화 해보자
    x = np.arange(iterations)
    for key, losses in train_losses.items():
        plt.plot(x, losses, label=key)
    plt.title('Losses')
    plt.xlabel('# of training')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()