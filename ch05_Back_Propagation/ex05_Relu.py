"""
ReLU(Rectified Linear Unit)
    relu(x) = x (if x > 0), 0 (otherwise) = max(0, x) : forward
    relu_prime(x) = 1 (if x > 0), 0 (otherwise) : backward

input x가 0보다 크면 x를, 0보다 작다면 0을 출력한다.
따라서, 다음 단계의 입력으로 float가 전달되는 Sigmoid 함수에 비해 Back Propagation에서 이전 단계에서 넘어온 값을 찾기가 쉽다.
"""
import numpy as np


class Relu:
    """
    ReLU(Rectified Linear Unit)
    relu(x) = x (if x > 0), 0 (otherwise) = max(0, x) : forward
    relu_prime(x) = 1 (if x > 0), 0 (otherwise) : backward
    """
    def __init__(self):
        # relu 함수의 input 값(x)가 0보다 큰 지 작은 지를 저장할 field
        self.mask = None # .mask: Ture/False로 구성된 numpy array. input <= 0 ~> True / input > 0 ~> False

    def forward(self, x):
        self.mask = (x <= 0)
        return np.maximum(0, x) # np.maximum(a, b) : a, b 둘 중 더 큰 값을 찾음

    def backward(self, dout):
        # print('masking 전:', dout)
        dout[self.mask] = 0
        # print('masking 후:', dout)
        dx = dout
        return dx


if __name__ == '__main__':
    # ReLU 객체를 생성
    relu_gate = Relu()

    # x = 1일 때 relu 함수의 리턴값 y
    y = relu_gate.forward(1)
    print('y =', y)

    np.random.seed(103)
    x = np.random.randn(5)
    print('x =', x)
    # x = [-1.24927835 -0.26033141  0.3837933  -0.38546147 -1.08513673]
    y = relu_gate.forward(x)
    # y = [0.        0.        0.3837933 0.        0.       ] ~~~> 음수는 0을, 양수는 그 값을 그대로 출력
    print('y =', y)  # relu 함수의 리턴 값
    print('mask =', relu_gate.mask)
    # relu_gate의 필드 mask. mask = [ True  True False  True  True]

    # back propagation(역전파)
    delta = np.random.randn(5)
    dx = relu_gate.backward(delta)







