"""
2차원 Convolution 연산

2차원 Convolution 연산에서의 input x는 2차원이므로 필터 w 역시도 2차원이어야한다.

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np


def convolution_2d(x, w):
    """
    x, w = 2d ndarray
    x.shape >= w.shape
    x와 w의 cross-correlation 계산(w를 flipping X)
    """
    # convolution 연산 결과 행렬 2d ndarray의 shape ~> (rows, cols)
    rows = x.shape[0] - w.shape[0] + 1 # xh - wh + 1
    cols = x.shape[0] - w.shape[1] + 1 # xh - ww + 1
    conv = []
    for i in range(rows):
        for j in range(cols):
            x_sub = x[i:i+w.shape[0], j:j+w.shape[1]]
            # x_sub = x[0:wh, 0:ww] / x[0:wh, 1:1+ww] / x[1:1+wh, 0:ww] / x[1:1+wh, 1:1+ww]
            fma = np.sum(x_sub * w)
            conv.append(fma)

    return np.array(conv).reshape((rows, cols))


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 10).reshape((3, 3))
    print('x =', x)
    w = np.array([[2, 0], [0, 0]])
    print('w =', w)

    # x의 높이(h, 세로, row 개수), 너비(w, 가로, col 개수)
    xh, xw = x.shape[0], x.shape[1]
    print('xh, xw =', xh, xw) # xh, xw = 3 3

    # w의 높이(h, 세로, row 개수), 너비(w, 가로, col 개수)
    wh, ww = w.shape[0], w.shape[1]
    print('wh, ww =', wh, ww) # wh, ww = 2 2

    x_sub1 = x[0:wh, 0:ww]
    print('x_sub1 =', x_sub1) # x_sub1 = [[1 2] [4 5]]
    fma1 = np.sum(x_sub1 * w)
    print('fma1 =', fma1) # fma1 = 2

    x_sub2 = x[0:wh, 1:1+ww]
    print('x_sub2 =', x_sub2) # x_sub2 = [[2 3] [5 6]]
    fma2 = np.sum(x_sub2 * w)
    print('fma2 =', fma2) # fma2 = 4

    x_sub3 = x[1:1+wh, 0:ww]
    print('x_sub3 =', x_sub3) # x_sub3 = [[4 5] [7 8]]
    fma3 = np.sum(x_sub3 * w)
    print('fma3 =', fma3) # fma3 = 8

    x_sub4 = x[1:1+wh, 1:1+ww]
    print('x_sub4 =', x_sub4) # x_sub4 = [[5 6] [8 9]]
    fma4 = np.sum(x_sub4 * w)
    print('fma4 =', fma4) # fma4 = 10

    conv = np.array([fma1, fma2, fma3, fma4]).reshape((2, 2))
    print('conv =', conv) # conv = [[ 2  4] [ 8 10]]

    cross_corr = convolution_2d(x, w)
    print('cross_corr =', cross_corr) # cross_corr = [[ 2  4] [ 8 10]]

    # 이번에는 x(5, 5)와 w(3, 3)를 랜덤하게 생성
    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    print('x =', x)
    print('w =', w)

    
