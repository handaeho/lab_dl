"""
Padding(패딩)
= Convolution 연산을 수행하기 전, 입력 데이터 주변을 특정 값으로 채우는 것.

패딩은 주로 '출력 크기를 조정할 목적'으로 사용한다.
합성곱 연산을 거칠때마다 데이터의 크기가 작아지면 어느 순간에는 크기가 1이 되어 더이상 합성곱 연산을 적용할 수 없게 된다.
이를 방지하기 위해 '패딩'을 사용한다.
패딩을 사용함으로써 입력 데이터의 공간적 크기를 고정한 채로 다음 layer에 전달할 수 있다.
"""
import numpy as np

if __name__ == '__main__':
    np.random.seed(113)

    # 1차원 ndarray
    x = np.arange(1, 6)
    print(x)

    x_pad = np.pad(x, # 패딩 넣을 배열
                   pad_width=1, # 패딩 크기
                   mode='constant', # 패딩에 넣을 숫자 타입
                   constant_values=0) # 상수(constant)로 지정할 패딩 값
    print(x_pad)

    x_pad = np.pad(x, pad_width=(2, 3), # (패딩 시작 부분 크기, 패딩 끝 부분 크기)
                   mode='constant', constant_values=0)
    print(x_pad)

    x_pad = np.pad(x, pad_width=2, mode='minimum')
    print(x_pad)

    # 2차원 ndarray
    x = np.arange(1, 10).reshape((3, 3))
    print(x)

    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print(x_pad)

    # axis=0 방향 before-padding = 1
    # axis=0 방향 after-padding = 2
    # axis=1 방향 before-padding = 1
    # axis=1 방향 after-padding = 2
    x_pad = np.pad(x, pad_width=(1, 2), mode='constant', constant_values=0)
    print(x_pad)

    # (1, 2) = (axis=0 before, axis=0 after)
    # (3, 4) = (axit=1 before, axix=1 after)
    x_pad = np.pad(x, pad_width=((1, 2), (3, 4)),
                   mode='constant', constant_values=0)
    print(x_pad)






