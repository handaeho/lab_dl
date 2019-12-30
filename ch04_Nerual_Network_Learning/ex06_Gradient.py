"""
Gradient (기울기) = '기울기'는 '각 지점에서 낮아지는 방향'이다.

즉,'기울기'가 가리키는 쪽은 '각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향'이다.
"""

import numpy as np
import matplotlib.pyplot as plt

from ch04_Nerual_Network_Learning.ex05_Differential import numerical_gradient


def fn(x):
    """ x = [x0, x1] """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1) # y축(열)기준 계산


if __name__ == '__main__':
    x0 = np.arange(-1, 2)
    x1 = np.arange(-1, 2)
    print(f'x0 = {x0}, x1 = {x1}')
    # x0 = [-1  0  1], x1 = [-1  0  1]

    X, Y = np.meshgrid(x0, x1)
    print('X =', X)
    # X = [[-1  0  1]
    #  [-1  0  1]
    #  [-1  0  1]]
    print('Y =', Y)
    # Y = [[-1 -1 -1]
    #  [ 0  0  0]
    #  [ 1  1  1]]
    # meshgrid(x, y) ~> X는 각 행이 x의 복사본인 행렬이고, Y는 각 열이 y의 복사본인 행렬.
    # 좌표 X와 Y로 표현되는 그리드에는 length(y)개의 행과 length(x)개의 열이 있다.

    X = X.flatten()
    Y = Y.flatten()
    print('X =', X)
    print('Y =', Y) # ~> 2차원 배열을 1차원으로 'flatten(평평하게)'
    # X = [-1  0  1 -1  0  1 -1  0  1]
    # Y = [-1 -1 -1  0  0  0  1  1  1]

    XY = np.array([X, Y])
    print('XY =', XY) # ~> 평평해진 1차원 배열 X, Y를 합쳐 다시 2차원 배열로
    # XY = [[-1  0  1 -1  0  1 -1  0  1]
    #  [-1 -1 -1  0  0  0  1  1  1]]

    grdients = numerical_gradient(fn, XY)
    print('grdients =', grdients)
    # grdients = [[-1  0  0  0  0  0  0  0  0]
    #  [-1 -1  0  0  0  0  0  0  0]] ~> XY의 각 값에 대한 미분 값

    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()
    XY = np.array([X, Y])
    grdients = numerical_gradient(fn, XY)
    print('grdients =', grdients)

    # 시각화
    plt.quiver(X, Y, -grdients[0], -grdients[1], angles='xy')
    # ~> quiver(x좌표, y 좌표, x 좌표의 해당 값, y 좌표의 해당 값, angles='벡터의 각도를 잡을 기준')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()