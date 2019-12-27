"""
Loss Function(손실 함수) - 오차 제곱 합(SSE, Sum of Squares for Error)
    Error = 1/2 * sum((예측값 - 실제값)^2)
"""
import pickle
import numpy as np

from ch03_Neural_Network.ex11_Mini_Batch import forward
from dataset.mnist import load_mnist


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)
    # ~> 'one_hot_label=True' -> one_hot_encoding을 True로 설정한다.
    # one_hot_encoding: 한 원소만 1로 나타내고, 나머지는 0으로 나타내는 것.

    y_true = y_test[:10]
    print('y_true[:10] =', y_true)

    with open('sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)

    y_pred = forward(network, X_test[:10])
    print('y_pred =', y_pred)

    print('y_true[0] =', y_true[0])
    print('y_pred[0] =', y_pred[0])

    print('y_true[8] =', y_true[8])
    print('y_pred[8] =', y_pred[8])
    print('sq_srr_[8] =', np.sum((y_true[8] - y_pred[8]) ** 2))

    # 오차
    error = y_pred[0] - y_true[0]
    print('error =', error)

    # 오차 제곱
    sq_err = error ** 2
    print('sq_error =', sq_err)

    # 평균 제곱 오차
    mse = np.mean(sq_err)
    print('mse =', mse)

    # 평균 제곱근 오차
    rmse = np.sqrt(mse)
    print('rmse =', rmse)
