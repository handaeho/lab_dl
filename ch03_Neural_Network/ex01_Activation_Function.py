"""
Perception
    - Input : x1, x2
    - Output : a = (x1 * w1) + (x2 + w2) + b
    ~~~> y = 0 (a <= 임계값) 또는 1 (a > 임계값)

신경망의 뉴런(Neuron)에서는 입력 신호의 가중치 합을 출력 값으로 변환해주는 함수가 존재.
이 함수를 '활성화 함수(Activation Function)'라고 한다.
"""
import math
import numpy as np
import matplotlib.pyplot as plt


# 1) 계단 함수
def step_function(x):
    """
    x <=0 : 0을 리턴
    x > 0 : 1을 리턴
    """
    # if x > 0:
    #     return 1
    # else:
    #     return 0
    y = x > 0 # 이때 y에는 배열 x를 0과 부등호 연산한 결과가 bool 배열(0보다 크면 True / 아니라면 False)로 저장됨.
    return y.astype(np.int) # 이렇게 만들어진 bool 배열 y를 int 타입으로 변환해서 True = 1, False = 0으로 리턴.


# 2) 시그모이드 함수
def sigmoid_function(x):
    """
    계단 함수의 0 또는 1과는 다르게 실수를 리턴한다.
    """
    return 1 / (1 + np.exp(-x))
    # math.exp VS np.exp
    # - math.exp(x) : x에 float 타입만 올 수 있다.
    # - np.exp(x) : x에 number, ndarray, iterable(list, tuple 등)타입 등이 올 수 있다.


# 3) ReLU 함수
def relu_function(x):
    """
    x <= 0 : 0을 리턴
    x > 0 : x를 리턴
    """
    return np.maximum(0, x) # np.maximum(a, b) : a, b 중 더 큰 값을 선택해 반환


if __name__ == '__main__':
    # 1) 계단(Step) 함수
    x = np.arange(-3.0, 4.0)
    print('x = ', x) # x =  [-3. -2. -1.  0.  1.  2.  3.]
    y = step_function(x)
    print('y = ', y) # y =  [0 0 0 0 1 1 1]

    # 2) Sigmoid 함수
    x = np.arange(-3.0, 4.0)
    y = sigmoid_function(x)
    print('y = ', y)

    # 3) ReLU 함수
    x = np.arange(-3.0, 4.0)
    y = relu_function(x)
    print('y = ', y)  # y =  [0. 0. 0. 0. 1. 2. 3.]

    # Step 함수, Sigmoid 함수, ReLU 함수의 그래프
    x = np.arange(-5.0, 5.0, 0.01)  # -5.0 <= x < 5.0 구간에서 step = 0.01
    y_step = step_function(x)
    y_sigmoid = sigmoid_function(x)
    y_relu = relu_function(x)
    plt.plot(x, y_step, label='Step Function')
    plt.plot(x, y_sigmoid, label='Sigmoid Function')
    plt.plot(x, y_relu, label='ReLU Function')
    plt.legend()
    plt.show()



