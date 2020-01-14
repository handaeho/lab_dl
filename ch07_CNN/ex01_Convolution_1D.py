"""
CNN(합성곱 신경망, Convolutional Neural Network)
= 이미지 인식과 음성 인식등 다양한 딥러닝 분야에서 사용.

1차원 Convolution(합성곱), Cross-Correlation(교차 상관) 연산
~> 합성곱 연산은 '필터의 Window(윈도우)'를 일정 간격으로 이동하며 입력 데이터에 적용한다.
   'dot 연산'이 아닌, 입력과 필터에 대응하는 원소끼리 곱하고, 그 총합을 구한다.(단일 곱셈 누산, FMA)
   그리고 그 결과를 출력의 해당 장소에 저장한다. 이 과정을 모든 장소에서 수행하면 합성곱 연산의 출력이 된다.
   필터를 반전(flipping)해서 적용하면 합성곱, 그렇지 않으면 교차상관 연산이다.

   사실 딥러닝에서는 이를 잘 구분하지 않으며,
   CNN에서는 가중치 행렬을 난수로 생성하고 gradient descent등을 사용해 갱신하기 때문에 대부분의 경우 '교차상관 연산'을 사용한다.

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np


def convolution_1d(x, w):
    """
    x, w = 1d ndarray, len(x) >= len(w)
    x와 w의 합성곱 연산
    """
    w_r = np.flip(w) # w를 반전(flipping)
    nx = len(x)
    nw = len(w)
    n = nx - nw + 1 # convolution 연산 결과의 원소 개수
    conv = []
    for i in range(n):
        x_sub = x[i:i+nw]
        fma = np.sum(x_sub * w_r)
        conv.append(fma)

    return np.array(conv)


def cross_correlation_1d(x, w):
    """
    x, w = 1d ndarray, len(x) >= len(w)
    x와 w의 교차상관 연산

    convolution_1d 함수를 cross_correlation_1d를 사용하도록 수정
    w를 flipping 하지 않는다.
    """
    nx = len(x)
    nw = len(w)
    n = nx - nw + 1  # convolution 연산 결과의 원소 개수
    conv = []
    for i in range(n) :
        x_sub = x[i :i + nw]
        fma = np.sum(x_sub * w)
        conv.append(fma)

    return np.array(conv)


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 6)
    print('x =', x) # x = [1 2 3 4 5]

    w = np.array([2, 1])
    print('w =', w) # w = [2 1]

    # Convolution(합성곱) 연산 ~> x Conv w

    # 1) w를 flipping(반전)
    # w_r = np.array([1, 2])
    w_r = np.flip(w)
    print('w_r =', w_r) # w_r = [1 2]

    # 2) FMA(Fused Multiply Add, 단일 곱셉 누산)
    for i in range(4):
        x_sub = x[i:i+2] #(0, 1), (1, 2), (2, 3), (3, 4)
        fma = np.dot(x_sub, w_r) # np.sum(x_sub * w_r) ~> 1차원에서는 dot연산을 사용해도 된다.
        print('fma =', fma, end=' ') # fma = 5 fma = 8 fma = 11 fma = 14
    print()

    # 1차원 convolution 연산 결과의 크기(원소의 개수) ~> len(x) - len(w) + 1
    # convolution_1d 함수 테스트
    conv = convolution_1d(x, w)
    print('conv =', conv) # conv = [ 5  8 11 14]

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv = convolution_1d(x, w)
    print('conv =', conv) # conv = [ 7 10 13]

    # Cross-Correlation(교차상관) 연산 ~> 합성곱 연산과는 다르게, 'w를 반전(flipping) 하지 않는다.'
    # CNN에서는 가중치 행렬을 난수로 생성하고 gradient descent등을 사용해 갱신하기 때문에 대부분의 경우 교차상관 연산을 사용한다.
    # Cross-correlation 함수 테스트
    cross_corr = cross_correlation_1d(x, w)
    print('cross_corr =', cross_corr) # cross_corr = [ 5  8 11]




