"""
DropOut(드롭아웃)
= 뉴런을 임의로 삭제하면서 학습하는 방법.
훈련시에 뉴런을 무작위로 삭제하며 학습하게 되고, 삭제된 뉴런은 신호를 전달하지 않는다.

훈련 할 때 데이터를 흘릴떄마다 삭제할 뉴런을 무작위로 선택하고 시험 할 때에는 모든 뉴런에 신호를 전달한다.
단, 시험할 때는 각 뉴런의 출력에 훈련할 때 삭제하지 않은 비율을 곱하여 출력한다.

(참고)
앙상블 학습(Ensemble Learning)
~> '개별적으로 학습시킨 여러 모델의 출력의 평균'을 최종 출력으로 삼는 방식.
예를 들어, 네트워크 5개를 따로따로 학습시키고, 시험시에는 그 5개의 출력에 대해 평균을 내어 답하는 것.

앙상블 학습은 드롭아웃과 유사하다.
드롭아웃이 학습 할 때 뉴런을 무작위로 삭제하는 행위를 매번 다른 모델을 학습시키는 것으로도 볼 수 있기 때문이다.
그리고 추론 시에는 뉴런의 출력에 삭제한 비율을 곱함으로써 앙상블 학습처럼 평균을 내는것과 같은 효과를 얻는다.
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06_Optimization.ex02_SGD import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist

np.random.seed(110)

# x = np.random.rand(20)  # 0.0 ~ 0.999999... 균등 분포에서 뽑은 난수
# print(x)
# mask = x > 0.5
# print(mask)
# print(x * mask)

# 데이터 준비
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# 신경망 생성
dropout_ratio = 0.1
neural_net = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100],
                                 output_size=10,
                                 use_dropout=True,
                                 dropout_ration=dropout_ratio)

X_train = X_train[:500]
Y_train = Y_train[:500]
X_test = X_test[:500]
Y_test = Y_test[:500]

epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 경우
mini_batch_size = 100  # 1번 forward에 보낼 데이터 샘플 개수
train_size = X_train.shape[0]
iter_per_epoch = int(max(train_size / mini_batch_size, 1))
# 학습하면서 학습/테스트 데이터의 정확도를 각 에포크마다 기록
train_accuracies = []
test_accuracies = []

optimizer = Sgd(learning_rate=0.01)  # optimizer

for epoch in range(epochs):
    indices = np.arange(train_size)
    np.random.shuffle(indices) # 무작위 선택을 위한 셔플(무작위 삭제는 무작위 선택과 같으므로)
    for i in range(iter_per_epoch):
        # 미니배치 사이즈만큼 무작위로 선택
        iter_idx = indices[(i * mini_batch_size):((i+1) * mini_batch_size)]
        x_batch = X_train[iter_idx]
        y_batch = Y_train[iter_idx]
        # 무직위로 선택된 대상들에 대해서만 기울기를 계산하고 파라미터 갱신
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
plt.title(f'Dropout (ratio={dropout_ratio})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()



