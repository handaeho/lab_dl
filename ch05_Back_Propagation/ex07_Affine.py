"""
Affine Transformation(어파인 변환) = 신경망의 순전파에서 수행하는 가중치 행렬의 dot 연산

'Affine 클래스'는 '가중치 행렬 W와 bias 행렬 b'를 '전단계의 입력'과 'dot 연산'하는 것을 '모두 하나로 묶어 표현'한다.
"""
import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W  # weight 행렬
        self.b = b  # bias 행렬
        self.X = None  # 입력 행렬을 저장할 field(변수)
        self.dW = None  # W 행렬 gradient -> W = W - lr * dW에서 사용.
        self.db = None  # b 행렬 gradient -> b  = b - lr * db에서 사용.

    def forward(self, X):
        self.X = X  # 나중에 역전파에서 사용됨.
        out = X.dot(self.W) + self.b

        return out

    def backward(self, dout):
        # b 행렬 방향으로의 gradient
        self.db = np.sum(dout, axis=0)
        # Z 행렬 방향으로의 gradient -> W방향, X방향으로의 gradient
        self.dW = self.X.T.dot(dout)
        # GD를 사용해서 W, b를 fitting시킬 때 사용하기 위해서.
        dX = dout.dot(self.W.T)

        return dX


if __name__ == '__main__':
    np.random.seed(103)
    X = np.random.randint(10, size=(2, 3))  # 입력 행렬

    W = np.random.randint(10, size=(3, 5))  # 가중치 행렬
    b = np.random.randint(10, size=5)  # bias 행렬

    affine = Affine(W, b)  # Affine 클래스의 객체 생성
    Y = affine.forward(X)  # Affine의 출력값. Y = X @ (W, b) ~~~> 각 X에 대한 Y = X @ W + b를 의미
    print('Y =', Y)

    dout = np.random.randn(2, 5)
    dX = affine.backward(dout)
    print('dX =', dX)
    print('dW =', affine.dW)
    print('db =', affine.db)


