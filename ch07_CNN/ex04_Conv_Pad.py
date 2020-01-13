"""
input_data에 Padding을 적용한 Convolution 연산
"""
import numpy as np

from scipy.signal import convolve, correlate, convolve2d, correlate2d
from ch07_CNN.ex01_Convolution_1D import convolution_1d


if __name__ == '__main__':
    x = np.arange(1, 6)
    print('x =', x) # x = [1 2 3 4 5]
    w = np.array([2, 0])
    print(convolution_1d(x, w)) # [4 6 8 10]
    # 일반적인 convolution(x, w)의 결과의 shape는 (4, )

    # 1) convolution 연산에서 x의 원소 중 '1'과 '5'는 연산에 한번만 기여를 하며, '2', '3', '4'는 2번씩 기여를 한다.
    # convolution 연산에서 x의 모든 원소가 '동일한 기여'를 할 수 있도록 padding.
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print(convolution_1d(x_pad, w)) # [2 4 6 8 10 0]

    # 2) convolution 연산에서 input x의 shape는 (5, )이다.
    # convolution 연산의 결과를 x와 '동일한 크기'로 다음 layer로 전달 할 수 있도록 padding.
    x_pad = np.pad(x, pad_width=(1, 0), mode='constant', constant_values=0)
    # ~> pad_width=(1, 0): before-padding만 설정
    print(convolution_1d(x_pad, w)) # [ 2  4  6  8 10]

    x_pad = np.pad(x, pad_width=(0, 1), mode='constant', constant_values=0)
    # ~> pad_width=(0, 1): after-padding만 설정
    print(convolution_1d(x_pad, w))  # [ 4  6  8 10  0]

    # convolution 연산을 위해 'scipy.signal.convolve() 함수'를 이용하면?
    conv = convolve(x, w, mode='valid') # ~> 일반적인 convolution 연산
    print('conv =', conv) # conv = [ 4  6  8 10]

    conv = convolve(x, w, mode='full') # ~> (1)번 방법과 같이 x의 모든 원소가 convolution 연산에 '동일한 기여'
    print('conv =', conv) # conv = [ 2  4  6  8 10  0]

    conv = convolve(x, w, mode='same') # ~> (2)번 방법과 같이 convolution 연산 결과가 x와 '동일한 크기'
    print('conv =', conv)  # conv = [ 2  4  6  8 10]

    # w를 flipping 하지 않은 Cross-Correlation 연산을 위해 scipy.signal.correlate() 함수를 이용하면?
    corr = correlate(x, w, mode='valid') # ~> 일반적인  cross-correlation 연산 연산
    print('corr =', corr) # corr = [2 4 6 8]

    corr = correlate(x, w, mode='full') # ~> (1)번 방법과 같이 x의 모든 원소가 cross-correlation 연산에 '동일한 기여'
    print('corr =', corr) # corr = [ 0  2  4  6  8 10]

    corr = correlate(x, w, mode='same')  # ~> (2)번 방법과 같이 cross-correlation 연산 결과가 x와 '동일한 크기'
    print('corr =', corr) # corr = [0 2 4 6 8]

    # convolution 연산 / cross-correlation 연산을 2차원 데이터에 적용
    x = np.array([[1, 2 ,3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]]) # input 2차원 ndarray x(4, 4)
    w = np.array([[2, 0, 1], [0, 1, 2], [1, 0, 2]]) # 2차원 ndarray w(3, 3)

    # scipy.signal.convolve2d() ~> convolution 연산을 2차원 데이터에 적용
    conv_2d = convolve2d(x, w, mode='valid')
    print('conv_2d =', conv_2d)
    conv_2d = convolve2d(x, w, mode='full')
    print('conv_2d =', conv_2d)
    conv_2d = convolve2d(x, w, mode='same')
    print('conv_2d =', conv_2d)

    # scipy.signal.correlate2d() ~> cross-correlation 연산을 2차원 데이터에 적용
    corr_2d = correlate2d(x, w, mode='valid')
    print('corr_2d =', corr_2d)
    corr_2d = correlate2d(x, w, mode='full')
    print('corr_2d =', corr_2d)
    corr_2d = correlate2d(x, w, mode='same')
    print('corr_2d =', corr_2d)

