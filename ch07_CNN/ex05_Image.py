"""
CNN(Convolution Neural Network, 합성곱 신경망)
= 이미지 인식과 음성 인식등 다양한 딥러닝 분야에서 사용.

완전연결(Fully-Connected) 신경망: 'Affine 계층'으로 구현
    input -> [Affine] -> [ReLU] -> [Affine] -> [ReLU] -> [Affine] -> [Softmax] -> output

CNN: 합성곱 계층(Convolutional Layer) & 폴링 계층(Pooling Layer) 추가.
    input -> [Conv] -> [ReLU] -> [Pooling] -> [Conv] -> [ReLU] -> [Pooling] -> [Affine] -> [Softmax] -> output
    output에 가까운 layer에서는 '[Affine] -> [ReLU]' 구성을 사용할 수 있다.
    그리고 마지막 layer에서는 '[Affine] -> [Softmax]' 구성을 그대로 사용한다.

CNN은 각 layer 사이에서 3차원 데이터 같은 '입체적인 데이터'가 흐른다는 것이 완전연결 신경망과 다르다.
완전연결 신경망은 3차원 입력 데이터를 1차원으로 평탄화해서 전달하기 때문에 입력의 특징을 제대로 살릴수 없다.
그러나 CNN은 3차원 입력데이터를 그대로 3차원으로 전달하기 떄문에 입력의 특징을 제대로 전달할 수 있다.

CNN에서의 데이터 = '특징 맵(Feature Map)'
CNN에서의 입력 데이터 = '입력 특징 맵(Input Feature Map)'
CNN에서의 출력 데이터 = '출력 특징 맵(Output Feature Map)'


CNN에서는 '필터의 매개변수'가 'Weight'에 해당된다. bias는 항상 하나(1x1)만 존재하고 필터를 적용한 모든 원소에 더한다.
    input -> Conv filter -> + bias -> output
"""
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import convolve, correlate


# jpg 파일 open
img = Image.open('sample.jpg', mode='r')
img_pixel = np.array(img)
print(img_pixel.shape) # (937, 1920, 3) ~> (세로길이(height), 가로길이(width), color-depth(RGB))

# Cf) 머신러닝의 라이브러리에 따라 color의 위치가 변경될 수 있다.
# Tensorflow: channel-last 방식. color-depth가 n차원 배열의 마지막 차원
# Theano:  channel-first 방식. color-depth가 n차원 배열의 첫번째 차원
# Keras: 두가지 방식 모두 지원.

# 이미지 화면 출력
plt.imshow(img_pixel) # pixel로 변화된 이미지를 전달해야 한다.
plt.show()

# 이미지의 RED/Green/Blue 값 정보
print(img_pixel[:, :, 0])
print(img_pixel[:, :, 1])
print(img_pixel[:, :, 2])

# 3x3x3 필터
filter = np.zeros((3, 3, 3))
print('filter =', filter)

# filter의 일부 값 수정
filter[1, 1, 0] = 255
print('filter =', filter)

# 이미지와 필터를 convolution 연산
transformed_conv = convolve(img_pixel, filter, mode='same') / 255
# ~> 0 ~ 1사이의 값으로 반들어 주기 위해 255로 나누고, input의 크기를 유지하기 위해 mode='same'
plt.imshow(transformed_conv)
plt.show()

# 이미지와 필터를 cross-correlation 연산
transformed_corr = correlate(img_pixel, filter, mode='same') / 255
plt.imshow(transformed_corr)
plt.show()





