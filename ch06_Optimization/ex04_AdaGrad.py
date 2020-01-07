"""
파라미터 최적화 알고리즘

3) AdaGrad(학습률 감소, Adaptive Gradient)
= SGD(W = W - lr * dL/dW)는 learning_rate가 고정되어 있다.
그런데 이때, 학습률이 너무 작으면 학습 시간이 너무 길고, 너무 크면 발산해서 학습이 제대로 이루어지지 않는다.
이런 문제를 AdaGrad 는 학습률 감소(learning rate decay) 를 통해 해결한다.
개별 파라미터에 '적응적인 학습률'을 조정하면서 학습을 진행한다.
처음에는 큰 학습률로 시작하고, 점점 학습률을 줄여 나가면서 파라미터를 갱신한다.

h <- h + dL/dW ⊙ dL/dW
lr = lr / sqrt(h)
W <- W - lr * 1/sqrt(h) * dL/dW

⊙: element wise multiplecation(행렬의 원소별 곱셈, dot 연산과 다르다. 위치가 대응되는 원소끼리 곱하는 것)
h에 이전 기울기의 제곱들이 누적되어 더해지게 되는데, parameter 를 업데이트 할때, 1/√h 를 통해 학습률을 조정한다.
이를 통해 parameter 중 많이 움직인 element 의 학습률이 낮아지는 효과를 만든다.
즉, 학습률의 감소가 element 마다 다르게 적용됨을 의미한다.

AdaGrad 알고리즘은 최소값을 향해 효율적으로 움직인다.
y축 방향은 기울기가 커서 처음에는 크게 움직이지만, 그 큰 움직임에 비례해 갱신 폭도 크다.
그래서 y축 방향으로 갱신 강도가 빠르게 약해지고, 움직임이 줄어든다.
"""
import numpy as np
import matplotlib.pyplot as plt

from ch06_Optimization.ex01_matplot3d import fn, fn_derivative


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = dict() # 각 파라미터(x, y 등)에 대한 미분값(기울기)을 저장할 dict

    def update(self, params, gradients):
        if not self.h: # dict 타입인 h가 비어있으면
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape의 영행렬 생성
                self.h[key] = np.zeros_like(params[key])

        # 속도 v와 파라미터 params를 갱신
        for key in params:
            # h <- h + dL/dW ⊙ dL/dW
            self.h[key] += gradients[key] * gradients[key]

            # W <- W - lr * 1/sqrt(h) * dL/dW
            params[key] -= self.lr * gradients[key] / (np.sqrt(self.h[key]) + 1e-7)
            # ~> 여기서 끝에 '1e-7'이라는 아주 작은 수를 더해줌으로써, '0으로 나누기를 방지'한다.


if __name__ == '__main__':
    # AdaGrad 클래스 객체(인스턴스)
    adagrad = AdaGrad(lr=1.5) # lr=1.5에서 시작해 조금씩 줄여나가보자.

    # update() 메소드 테스트

    # 신경망에서 찾고자 하는 파라미터의 초기값(-7, 2)
    params = {'x': -7., 'y': 2.}

    # 각 파라미터에 대한 변화율(gradient) 초기값
    gradients = {'x': 0., 'y': 0.}

    # 각 파라미터(x, y)를 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []

    # 파라미터(x, y)를 gradient에 따라 30번 adagrad.update()하고 변경된 값들을 리스트에 추가
    for i in range(30) :
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 현재 파라미터 위치의 gradient 계산
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        # 파라미터 갱신
        adagrad.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')
        # (-7.0, 2.0)
        # (-5.500000214285683, 0.5000000374999991)
        # (-4.573267968164272, 0.13619658308878757)
        # ...
        # (-0.17493063754566132, 1.2073244099652555e-15)
        # (-0.1543462534762518, 3.309322292911385e-16)
        # (-0.13618539990739692, 9.070978726153172e-17)

    # contour(등고선)그래프에 파라미터 갱신값 그래프 추가
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

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()