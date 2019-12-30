"""
Gradient Descent(경사 하강법) = '기울기'를 사용해 '함수의 최소값'을 찾는 방법.

현재 위치에서 기울어진 방향으로 일정 거리를 움직여 기울기를 구하고, 또 나아가며 기울기를 구해
함수의 최소값이 되게 값을 점차 줄여나간다.

-> x_new = x = lr(learning_rate) * df/dx ~~~> 이 과정을 반복(step_size 만큼)해 f(x)의 최소값을 찾는다.

'최소값'을 찾게 되면 '경사 하강법' / '최대값'을 찾게되면 '경사 상승법'이라고 하는데,
손실 함수의 부호를 반전시키면 최대값과 최소값은 같으므로, 이 둘은 본질적으로 같은 문제가 된다.

일반적으로 신경망 분야에서의 경사법은 '경사 하강법'을 의미한다.
"""
import numpy as np
import matplotlib.pyplot as plt

from ch04_Nerual_Network_Learning.ex05_Differential import numerical_gradient


def gradient_method(fn, x_init, lr=0.01, step=100):
    """
    기울기, 학습률과 반복 횟수에 따라 점차 변화시켜 f(x)의 최소값이 되는 x를 찾는다.

    :param fn: 함수 f(x)
    :param x_init: x의 초기 값
    :param lr: learning rate(학습률)
    :param step: 반복 횟수
    :return: f(x)의 최소값 x
    """
    x = x_init # 점차 변화시킬 변수
    x_history = [] # 변수의 변화 과정을 저장할 배열

    for i in range(step): # step 횟수만큼 반복
        x_history.append(x.copy()) # x의 복사본을 변화 과정 배열에 append
        # Cf) copy를 하는 이유?
        # 사실 변수 x 자체에는 값이 아닌 저장된 값이 위치하는 '주소'가 저장되는 것이다.
        # 따라서, copy를 해야 x의 변화 값들이 배열에 append.
        # 안하면 해당 주소에 덮어 씌워지며 계산된 최종 변화값만 append된다.
        grad = numerical_gradient(fn, x) # x에서의 gradient 계산
        x -= lr*grad # x_new = x = lr(learning_rate) * df/dx

    return x, np.array(x_history) # f(x)의 최소값 x와 변화되는 과정


def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    # gradient_method() 테스트
    init_x = np.array([4.0]) # x의 초기값
    x, x_history = gradient_method(fn, init_x, lr=0.1)
    print('x =', x)
    print('x_history =', x_history)
    # 학습률이 너무 작으면, 최소값을 찾는 시간이 너무 오래 걸리며,
    # 학습률이 너무 크면, 최소값을 찾지 못하고 발산하는 경우가 발생할 수 있다.

    init_x = np.array([4.0, -3.0])
    x, x_history = gradient_method(fn, init_x, lr=0.1, step=100)
    print('x =', x)
    print('x_history =', x_history)

    # x_history(최소값을 찾는 과정)를 산점도 그래프로 시각화
    plt.scatter(x_history[:, 0], x_history[:, 1])
    # '동심원' 형태로 보조선 그리기
    for r in range(1, 5):
        r = float(r) # 정수 타입을 실수 타입으로
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)
        y_pts2 = -np.sqrt(r**2 - x_pts**2)
        plt.plot(x_pts, y_pts1, ':', color='gray')
        plt.plot(x_pts, y_pts2, ':', color='gray')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')
    plt.axhline(color='0.8')
    plt.show()



