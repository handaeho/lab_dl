"""
Machine Learning(기계 학습) ~> Deep Learning(심층 학습)
training data set(학습 세트) / test data set(검증 세트)

신경망 층(layer)을 지나갈 때 사용되는 Weight(가중치) 행렬 & bias(편향) 행렬을 찾는것이 목적

'오차'를 최소화 하는 가중치 행렬을 찾는다. ~~~> loss(손실) 함수 / cost(비용) 함수의 값을 최소화 하는 가중치 행렬을 찾는다.

손실 함수
    1) 평균 제곱 오차(MSE, Mean Squared Error)
    2) 교차 엔트로피(Cross-Entropy)
"""
import numpy as np

from dataset.mnist import load_mnist


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()
    # 10개 테스트 데이터 이미지들의 실제 값
    print('y_true[:10] =', y_true[:10]) # y_true[:10] = [7 2 1 0 4 1 4 9 5 9]

    # 10개 테스트 데이터 이미지들의 예측 값이 예를 들어, 다음과 같다고 하자.
    y_pred = np.array([7, 2, 1, 6, 4, 1, 4, 9, 6, 9])
    print('y_pred =', y_pred) # y_pred = [7 2 1 6 4 1 4 9 6 9]

    # 오차(Error) ~> 각 항목별 차이
    error = y_pred - y_true[:10]
    print('error =', error) # error = [0 0 0 6 0 0 0 0 1 0]

    # 오차 제곱(Squared Error)
    sq_err = error ** 2
    print('squared error =', sq_err) # squared error = [ 0  0  0 36  0  0  0  0  1  0]

    # 평균 제곱 오차(MSE, Mean Squared Error)
    mse = np.mean(sq_err)
    print('mse =', mse) # mse = 3.7

    # 평균 제곱근 오차(RMS(=RMSE), Root Mean Squared Error)
    rmse = np.sqrt(mse)
    print('rmse =', rmse) # rmse = 1.9235384061671346



