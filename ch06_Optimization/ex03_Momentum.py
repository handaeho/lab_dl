"""
파라미터 최적화 알고리즘

2) Momentum(모멘텀)
= 공이 곡면을 따라 굴러간다고 생각해보자.
기울기가 완만하면 공의 속도가 비교적 느리고, 기울기가 급하면 공의 속도가 비교적 빨라진다.
이와 같은 개념으로 기울기가 커지면 속도가 빨라지고, 파라미터를 더 빠르게 하강시키게 되면서 최저점을 효율적으로 찾을 수 있다.

v는 일종의 가속도(혹은 속도) 같은 개념으로 생각하면 편하다.
v의 영향으로 인해 가중치가 감소하던 (혹은 증가하던) 방향으로 더 많이 변화하게 되는 것이다.


v: 속도(velocity)
m: 모멘텀 상수(momentum constant)
lr: 학습률(learning_rate)

v = m * v - lr * dL/dW
이를 가중치에 적용하면, W = W + v = W + m * v - lr * dL/dW
"""
import numpy as np
import matplotlib.pyplot as plt

from ch06_Optimization.ex01_matplot3d import fn_derivative, fn


class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr # learning_rate
        self.m = m # 모멘텀 상수(속도 v에 곱해줄 상수)
        self.v = dict() # 속도(각 파라미터 방향의 속도 저장({방향:속도})

    def update(self, params, gradients):
        if not self.v: # dict 타입의 v가 비어있으면
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape의 영행렬 생성
                self.v[key] = np.zeros_like(params[key])

        # 속도 v와 파라미터 params를 갱신
        for key in params:
            # v = m * v - lr * dL/dW
            # self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            self.v[key] *= self.m
            self.v[key] -= self.lr * gradients[key]

            # W = W + v = W + m * v - lr * dL/dW
            params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 클래스 객체(인스턴스)
    momentum = Momentum(lr=0.068212) # lr을 0.068212로 설정

    # update() 메소드 테스트

    # 신경망에서 찾고자 하는 파라미터의 초기값(-7, 2)
    params = {'x': -7., 'y': 2.}

    # 각 파라미터에 대한 변화율(gradient) 초기값
    gradients = {'x': 0., 'y': 0.}

    # 각 파라미터(x, y)를 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []

    # 파라미터(x, y)를 gradient에 따라 30번 momentum.update()하고 변경된 값들을 리스트에 추가
    for i in range(30) :
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')
        # (-7.0, 2.0)
        # (-6.9522516, 1.727152)
        # (-6.8618553413860806, 1.245963815552)
        # ...
        # (-0.8433373078846071, -0.32690092124591935)
        # (-0.6771750618471759, -0.16386536277743693)
        # (-0.5230098938816159, 0.005221808095746322)

    # 파라미터 x, y에 대한 함수 f(x, y)의 변화 모습을 등고선으로 시각화
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # contour(등고선)그래프에 파라미터 갱신값 그래프 추가
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()

