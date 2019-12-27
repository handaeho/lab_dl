"""
Loss Function(손실 함수) - 교차 엔트로피 오차(CEE, Cross Entropy Error)
    Entropy = -sum(실제값 * log(예측값))
"""
import pickle
import numpy as np

from ch03_Neural_Network.ex11_Mini_Batch import forward
from dataset.mnist import load_mnist


def _cross_entropy(y_pred, y_true):
    """ Entropy = -sum(실제값 * log(예측값)) """
    delta = 1e-7
    # ~> 'log0'이면 '-inf(-무한대)'가 되기 때문에, 이를 방지하기 위하여 아주 작은 값인 'delta'를 더해준다.

    return -np.sum(y_true * np.log(y_pred+delta))


def cross_entropy(y_pred, y_true):
    """ N차원 리스트 데이터에 대한 교차 엔트로피 오차의 '평균'을 구하여 보다 통일된 지표로 사용 """
    if y_pred.ndim == 1: # 예측 데이터가 1차원 리스트라면
        cee = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2: # 예측 데이터가 2차원 리스트라면
        cee = _cross_entropy(y_pred, y_true) / len(y_pred)
        # 'len(y_pred)'로 나누어 '엔트로피의 평균' 계산.

    return cee


if __name__ == '__main__':
    (T_train, y_train), (T_test, y_test) = load_mnist(one_hot_label=True)
    # one_hot_encoding: 한 원소만 1로 나타내고, 나머지는 0으로 나타내는 것.

    # 실제 값 y_true
    y_true = y_test[:10]

    with open('sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)

    # 에측 값 y_pred
    y_pred = forward(network, T_test[:10])
    print('y_true[0] =', y_true[0])
    # y_true[0] = [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    # ~> 실제로는 숫자 '7' 이미지
    print('y_pred[0] =', y_pred[0])
    # y_pred[0] = [8.4412488e-05 2.6350631e-06 7.1549456e-04 1.2586268e-03 1.1727943e-06
    #  4.4990851e-05 1.6269286e-08 9.9706501e-01 9.3744702e-06 8.1831118e-04]
    # ~> 예측은 숫자 '7'인 확률이 가장 크다고 판단.

    print('y_true[8] =', y_true[8])
    # y_true[8] = [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    # ~> 실제로는 숫자 '5' 이미지
    print('y_pred[8] =', y_pred[8])
    # y_pred[8] = [1.8853788e-03 3.4948647e-05 5.7895556e-03 6.6576863e-06 3.3810724e-02
    #  7.3766336e-03 9.4999015e-01 8.5003812e-06 1.0030542e-03 9.4286152e-05]
    # ~> 예측은 숫자 '8'인 확률이 가장 크다고 판단.

    # 교차 엔트로피 오차(CEE)
    cee1 = cross_entropy(y_pred[0], y_true[0])
    print('CEE =', cee1) # CEE = 0.00293918838724494 ~> 예측이 맞기때문에 엔트로피가 비교적 작다.

    cee2 = cross_entropy(y_pred[8], y_true[8])
    print('CEE =', cee2) # CEE = 4.909424304962158 ~> 예측이 틀렸기때문에 엔트로피가 비교적 크다.

    # 2차원 데이터의 교차 엔트로피 오차(CEE)
    print('CEE Mean =', cross_entropy(y_pred, y_true)) # CEE Mean = 0.5206955424044282
    # ~> 데이터의 인덱스를 따로 지정해 주지 않고, 전체 데이터에 대한 예측과 실제 값의 CEE

    # 만약 실제 값과 예측 값이 다음과 같다고 하자.
    # 단, y_pred / y_true가 'one_hot_encoding' 형태가 아니라면, 'one_hot_encoding' 형태로 변환 후 'Cross-Entropy'를 계산
    np.random.seed(1227)
    y_true = np.random.randint(10, size=10)
    print('y_true =', y_true)

    y_true_2 = np.zeros((y_true.size, 10)) # y_true를 (행 수=y_true의 사이즈, 열 수=10)의 '영 행렬'로 변환
    print(y_true_2)

    for i in range(y_true.size):
        y_true_2[i][y_true[i]] = 1 # 영행렬 y_true_2의 특정 값만 1로 변환해 'one_hot_encoding' 형태로 생성
    print(y_true_2)





