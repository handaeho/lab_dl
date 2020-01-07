"""
신경망의 학습 목적? 손실 함수의 값을 가능한 낮추는 파라미터를 찾는 것.

파라미터(Parameter)
    - 파라미터: Weight(가중치), bias(편향)
    - 하이퍼 파라미터: learning_rate(학습률), epoch, batch_size,
                     신경망의 뉴런 개수, 신경망의 hiddin_layer 개수

파라미터를 갱신하는 방법?  SGD, Momentum, AdaGrad, Adam

파라미터 최적화 알고리즘

1) SGD(확률적 경사 하강법, Stochastic Gradient Descent)
= W <- W - lr * dL/dW

W: 갱신할 가중치 파라미터
lr: learning_rate
dL/dW: W에 대한 손실함수의 기울기
'<-': 우변의 값으로 좌변의 값을 갱신한다.
"""
import numpy as np
import matplotlib.pyplot as plt

from ch06_Optimization.ex01_matplot3d import fn_derivative, fn


class Sgd:
    """SGD: Stochastic Gradient Descent(확률적 경사 하강법)
    W = W - lr * dL/dW
    W: 파라미터(가중치, 편향)
    lr: 학습률(learning rate)
    dL/dW: 변화율(gradient)
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """파라미터 params와 변화율 gradients가 주어지면,
        파라미터들을 갱신하는 메소드.
        params, gradients: 딕셔너리. {key: value, ...}
        """
        for key in params:
            # W = W - lr * dL/dW
            params[key] -= self.learning_rate * gradients[key] # W = W - lr * dL/dW


if __name__ == '__main__':
    # sgd 클래스 객체(인스턴스)
    sgd = Sgd(learning_rate=0.95)

    # ex01.py에서 작성한 함수 fn(x, y)의 최소값을 임의의 점에서 시작해 찾아감.
    init_position = (-7, 2)

    # 신경망에서 찾고자 하는 파라미터의 초기값
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]

    # 각 파라미터에 대한 변화율(gradient)
    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0

    # 각 파라미터(x, y)를 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []

    # 파라미터(x, y)를 gradient에 따라 30번 sgd.update()하고 변경된 값들을 리스트에 추가
    # x = x - lr * df/dx
    # y = y - lr * df/dy
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        sgd.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')
        # (-7, 2)
        # (-6.335, -1.7999999999999998)
        # (-5.733175, 1.6199999999999997)
        # ...
        # (-0.4727262060886904, -0.11629947400607987)
        # (-0.42781721651026483, 0.10466952660547188)
        # (-0.3871745809417897, -0.09420257394492468)

    # 파라미터 x, y에 대한 함수 f(x, y)의 변화 모습을 등고선으로 시각화
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 등고선 그래프에 파라미터(x, y)들이 갱신되는 과정을 추가.
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()