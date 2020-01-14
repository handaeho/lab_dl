"""
3차원 Convolution 연산

3차원 Convolution 연산에서의 input x는 3차원이므로 필터 w 역시도 3차원이어야 한다.

3차원 데이터는 특징 맵에서 '가로', '세로'에 추가로 '길이'를 갖게 된다.

주의할 점은 '입력 데이터의 채널 수'와 '필터의 채널 수'가 같아야 한다.
필터의 크기는 원하는 값으로 설정이 가능하지만, 모든 채널의 필터는 같은 크기여야 한다.

3차원 Convolution 연산은 데이터와 필터를 '직육면체 블록'으로 생각하면 쉽다.

입력 데이터(C, H, W), 필터(C, FH, FW)
C: channel / H: height / W: width / FH: filter height / FW: filter width
~> 필터를 FN개 적용하면, 출력 맵도 FN개가 생성된다.
그리고 그 FN개의 맵을 모으면 (FN, OH, OW)인 블록이 되며 이 블록을 다음 layer로 넘기는 것이 3차원 Convolution 연산이다.

3차원 Convolution 연산에서는 필터의 수도 고려를 해야하며,
필터의 가중치 데이터는 4차원을 갖는다. (출력채널수, 입력채널수, 높이, 너비)

bias는 채널 하나에 값 하나씩으로 구성된다.
형상이 다른 블록의 덧셈은 numpy의 'broadcast()'를 사용하면 된다.

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np
from scipy.signal import correlate


def convolution3d(x, y):
    """
    x, w = 3d ndarray
    x.shape = (c, h, w)
    y.shape = (c, fh, fw)

    h >= fh, w >= fw라고 가정

    x와 y의 cross-correlation 계산
    """
    h, w = x.shape[1], x.shape[2] # 입력 데이터 x의 height(h), width(w)
    fh, fw = y.shape[1], y.shape[2] # 필터 y의 filter height(fh), filter width(fw)
    oh = h - fh + 1 # 출력 데이터 output height(oh) ~> row 개수
    ow = w - fw + 1 # 출력 데이터 output width(ow) ~> col 개수
    conv = []
    for i in range(oh): # 출력 데이터
        for j in range(ow):
            x_sub = x[:, i:i+fh, j:j+fw] # 데이터와 필터의 채널 수는 동일하므로 높이와 너비에 대해 합성곱 연산
            fma = np.sum(x_sub * y)
            conv.append(fma)

    return np.array(conv).reshape((oh, ow)) # 출력 데이터의 형상에 맞게 reshape


if __name__ == '__main__':
    np.random.seed(114)

    # (3, 4, 4) shape의 3차원 ndarray ~~~> 3차원 입력 데이터
    x = np.random.randint(10, size=(3, 4, 4))
    print('x =', x)

    # (3, 3, 3) shape의 3차원 ndarray ~~~> 3차원 필터
    w = np.random.randint(5, size=(3, 3, 3))
    print('w =', w)

    # 3차원 입력 데이터와 3차원 필터의 convolution 연산(cross-correlation 연산)
    conv1 = correlate(x, w, mode='valid')
    print('conv1 =', conv1) # conv1 = [[[317 287] [308 286]]]

    # convolution3d() 함수 테스트
    fn_conv = convolution3d(x, w)
    print('fn_conv =', fn_conv) # fn_conv = [[317 287] [308 286]]

    x = np.random.randint(10, size=(3, 28, 28))
    y = np.random.rand(3, 16, 16)
    print('x =', x)
    print('filter y =', y)
    # ~> 데이터 x와 필터 y를 합성곱 연산한 출력은 13x13의 크기가 될것.

    conv2 = correlate(x, y, mode='valid') # scipy.correlation() 함수 사용
    fn_conv2 = convolution3d(x, y) # 직접 구현한 함수 사용
    print('conv2 =', conv2)
    print('fn_conv2 =', fn_conv2)
    print('conv2 shape =', conv2.shape) # conv2 shape = (1, 13, 13)
    print('fn_conv2 shape =', fn_conv2.shape) # fn_conv2 shape = (13, 13)




