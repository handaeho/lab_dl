"""
X -> [Affine] -> [ReLU] -> [Affine] -> [ReLU] -> [Affine] -> [SoftmaxwithLoss] -> [CEE] -> L

출력 L의 Activation Function(활성화 함수)으로 'Softmax 함수'를 적용하고,
이에 대한 Loss function(손실 함수)로는 'Cross Entropy Error(CEE)'를 사용하기 위해서 이를 클래스로 만든다.
그리고 순전파에서 loss를, 역전파에서 입력에 대한 미분값을 구한다.
"""
import numpy as np
from ch03_Neural_Network.ex11_Mini_Batch import softmax
from ch04_Neural_Network_Learning.ex03_CEE import cross_entropy


class SoftmaxWithLoss:
    def __init__(self):
        self.y_true = None  # 정답 레이블을 저장하기 위한 field(변수). one-hot-encoding
        self.y_pred = None  # softmax 함수의 출력(예측 레이블)을 저장하기 위한 field.
        self.loss = None  # cross_entropy 함수의 출력(손실, 오차)를 저장하기 위한 field.

    def forward(self, X, Y_true):
        self.y_true = Y_true
        self.y_pred = softmax(X)
        self.loss = cross_entropy(self.y_pred, self.y_true)
        return self.loss

    def backward(self, dout=1):
        if self.y_true.ndim == 1:  # 1차원 ndarray
            n = 1
        else:  # 2차원 ndarray
            n = self.y_true.shape[0]  # one-hot-encoding 행렬의 row 개수
        dx = (self.y_pred - self.y_true) / n  # 오차들의 평균((예측값 - 실제값) / 개수)
        return dx


if __name__ == '__main__':
    np.random.seed(103)

    x = np.random.randint(10, size=3)
    print('x =', x)

    y_true = np.array([1., 0., 0.])  # one-hot-encoding
    print('y =', y_true)

    swl = SoftmaxWithLoss()  # SoftmaxWitLoss 클래스 객체 생성
    loss = swl.forward(x, y_true)  # forward propagation
    print('y_pred =', swl.y_pred)
    print('loss =', loss)

    dx = swl.backward()  # back propagation(역전파)
    print('dx =', dx)

    print()  # 손실(loss)가 가장 큰 경우
    y_true = np.array([0, 0, 1])
    loss = swl.forward(x, y_true)
    print('y_pred =', swl.y_pred)
    print('loss =', loss)
    print('dx =', swl.backward())

    print()  # 손실(loss)가 가장 작은 경우
    y_true = np.array([0, 1, 0])
    loss = swl.forward(x, y_true)
    print('y_pred =', swl.y_pred)
    print('loss =', loss)
    print('dx =', swl.backward())





