"""
어떤 점 x에서의 함수 fn의 미분값

1) Numerical Differential(수치 미분)
= 함수 fn과 점 x가 주어졌을 때, 점 x에서의 함수 fn의 미분(도함수)값
    f'(x) = lim (f(x+h) - f(x)) / ((x+h) - x)

2) 중심차분(중앙차분)
수치 미분은 '점 x의 기울기'. 즉, '진정한 미분'과는 정확히 같지 않다. 그래서 수치 미분에는 오차가 포함된다.
따라서 오차를 줄이고자 점 x를 기준으로 그 전,후의 차분을 계산하는 '중심차분(중앙차분)'을 계산하자.
    f'(x) = lim (f(x+h) - f(x)) / ((x+h) - x) = lim (f(x+h) - f(x-h)) / ((x+h) - (x-h)) = lim (f(x+h) - f(x-h)) / 2h

3) Partial Differential(편미분) = 변수 1개를 상수 취급하고 나머지 변수에 대해 미분
x,y에서 각각에 대해 미분하면,
    df/dx = f'(x) = lim f((x+h), y) - f((x-h), y) / (x+h) - (x-h)
    df/dy = f'(y) = lim f(x, (y+h)) - f(x, (y-h)) / (y+h) - (y-h)

예를 들어, f(x, y) = x^2 + xy + y^2일 때,
    x에 대하여 편미분 하면 y를 상수 취급하여, df/dx = 2x + y
    y에 대하여 편미분 하면 x를 상수 취급하여, df/dy = x + 2y
"""
import numpy as np


def numerical_diff(fn, x):
    """
    Numerical Differential(수치 미분)을 개선한 '중심 차분'
    """
    h = 1e-4
    return (fn(x + h) - fn(x - h)) / (2 * h)


def _numerical_gradient(fn, x):
    """
    독립 변수 n개를 갖는 함수 fn에 대한 편미분

    :param fn: fn = fn(x0, x1, ..., xn)
    :param x: x = [x0, x1, ..., xn]
    :return: fn의 각 편미분 값들의 배열
    """
    h = 1e-4
    x = x.astype(np.float, copy=False) # x는 실수 타입이 되어야 한다.('copy=False'는 원본 데이터의 타입 자체를 변화)
    gradient = np.zeros_like(x) # 이는 np.zeros(x.shape)와 같음(x의 행, 열 크기와 같은 영행렬)

    for i in range(x.size): # 점 x의 리스트 사이즈만큼(전체 원소 개수만큼)
        ith_val = x[i] # i번째 value = x의 i번쨰 value
        x[i] = ith_val + h # f(x+h)
        fh1 = fn(x)
        x[i] = ith_val - h # f(x-h)
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2 * h) # 편미분 값을 계산해 각 위치에 대응하는 gradient 행렬 값 변경
        x[i] = ith_val # 한 변수에 대해 미분이 끝나고, 다음 변수에 대한 미분 계산을 위해 원래의 값으로 복원

    return gradient


def numerical_gradient(fn, x):
    """ x = [[x11, x12, x13, ...] ,
            [x21, x22, x23, ...],
            [x31, x32, x33, ...]]  """
    if x.ndim == 1: # x가 1차원 배열이라면
        return _numerical_gradient(fn, x)
    else: # x가 2차원 배열이라면
        grads = np.zeros_like(x) # x의 행, 열 크기와 같은 영행렬 생성
        for i, x_i in enumerate(x): # x의 index와 value를 각각 i, x에
            grads[i] = _numerical_gradient(fn, x_i) # 편미분 계산후, 각 위치에 결과 값
        return grads


def f1(x):
    """
    변수가 1개인 함수 f1
    """
    return 0.001 * x**2 + 0.01 * x


def f2(x):
    """
    변수가 2개인 함수 f2
    단, x = [x0, x1]
    """
    return np.sum(x**2) # 또는 x[0]**2 + x[1] **2


def f3(x):
    """
    변수가 3개인 함수 f3 = x0 + x1**2 + x2**3
    단, x = [x0, x1, x2]
    """
    return x[0] + x[1]**2 + x[2]**3


def f4(x):
    """
    변수가 2개인 함수 f4 = x0**2 + x0 * x1 + x1**2
    단, x = [x0, x1]
    """
    return x[0]**2 + x[0]*x[1] + x[1]**2


def f1_prime(x):
    """
    근사값을 사용하지 않은 함수 fn의 도함수
    """
    return 0.002 * x + 0.01


if __name__ == '__main__':
    # 변수가 1개인 함수 f1
    estimate = numerical_diff(f1, 3)
    print('근사값 =', estimate) # 근사값 = 0.016000000000043757
    real = f1_prime(3)
    print('실제값 =', real) # 실제값 = 0.016

    # 변수가 2개인 함수 f2 ~> '편미분'. 즉, 변수 1개를 상수 취급하고 나머지 변수에 대해 미분
    # 함수 f2의 점(3, 4)에서의 편미분
    estimate_1 = numerical_diff(lambda x: x**2 + 4**2, x=3)
    # ~> x0 = 3, x1 = 4일 때, x1에는 4를 대입하고(상수 취급), x0 = 3일때에 대해서 미분을 한다.
    print('estimate_1 =', estimate_1) # estimate_1 = 6.00000000000378

    estimate_2 = numerical_diff(lambda x: 3**2 + x**2, x=4)
    # ~> x0 = 3, x1 = 4일 때, x0에는 3를 대입하고(상수 취급), x1 = 4일때에 대해서 미분을 한다.
    print('estimate_2 =', estimate_2) # estimate_2 = 7.999999999999119

    # numerical_gradient() 함수를 이용한 편미분 계산
    gradient = numerical_gradient(f2, np.array([3., 4.]))
    print('gradient =', gradient) # gradient = [6. 8.] ~> (x0=3, x1=4)에서 기울기는 각각 6, 8

    # 변수가 3개인 함수 f3의 편미분 계산(x0 = 1, x1 = 1, x2 = 1)
    gradient_2 = numerical_gradient(f3, np.array([1., 1., 1.]))
    print('gradient_2 =', gradient_2) # gradient_2 = [1. 2. 3.00000001]

    # 변수가 2개인 함수 f4의 편미분 계산(x0 = 1, x1 = 2)
    gradient_3 = numerical_gradient(f4, np.array([1., 2.]))
    print('gradient_3 =', gradient_3) # gradient_3 = [4. 5.]
