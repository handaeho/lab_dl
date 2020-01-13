"""
Hyper Parameter(하이퍼 파라미터)
= 각 층의 뉴런 수, 배치 크기, 파라미터(W/b) 갱신시의 학습률, 가중치 감소 등

이러한 하이퍼 파라미터의 값을 적절하게 설정해야만 모델의 좋은 성능을 기대할 수 있다.

학습 데이터와 시험 데이터를 나누어 모델의 성능 평가 시, 시험 데이터에만 하이퍼 파라미터를 적용하면?
~> 시험 데이터에만 의존해 하이퍼 파라미터가 설정되므로, 시험 데이터 이외에는 적합하지 않게 된다.
즉, 시험데이터에 대해 '오버피팅'이 발생한다.

따라서 하이퍼 파라미터를 조정하기 위한 별도의 전용 확인 데이터인 '검증 데이터'가 필요하다.

학습(훈련) 데이터: 파라미터 학습
검증 데이터: 하이퍼 파라미터 성능 평가
시험 데이터: 신경망의 범용 성능 평가(이상적으로는 마지막에 한번만 사용)

하이퍼 파라미터 최적화의 핵심은 하이퍼 파라미터의 '최적 값'이 존재하는 범위를 조금씩 줄여나가는 것이다.
이때는 그리드 서치 같은 규칙적인 탐색보다는 '무작위로 샘플링해 탐색'하는것이 더 좋은 결과를 낸다.
왜냐하면, 최종 정확도에 미치는 영향력이 하이퍼 파라미터별로 조금씩 다르기 때문이다.

0단계: 하이퍼 파라미터 값의 범위 설정
1단계: 설정된 범위에서 하이퍼 파라미터 값을 무작위 선별
2단계: 1단계에서 샘플링한 하이퍼 파라미터 값을 사용해 학습하고, 검증 데이터로 정확도 평가(단, epoch는 작게)
3단계: 1단계와 2단계를 특정 횟수만큼 반복하며 그 정확도를 보고, 하이퍼 파라미터의 범위를 좁힌다.

참고)
하이퍼 파라미터의 최적화 할 때, '베이즈 최적화'라는 기법을 사용하기도 한다.
베이즈 최적화 ~> 베이즈 정리를 중심으로 최적화해 더 엄밀하고 효율적으로 수행.
"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': lr},
                      verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()
