"""
Image Index(이미지 인덱스)

머신러닝의 라이브러리에 따라 대상의 인덱스를 맞춰 주어야 올바르게 작동한다.

- Tensorflow: channel-last 방식. color-depth가 n차원 배열의 마지막 차원
- Theano:  channel-first 방식. color-depth가 n차원 배열의 첫번째 차원
- Keras: 두가지 방식 모두 지원.

np.moveaxis(A, x, y) ~> A의 x번 축을 y번 축으로 바꾼다.
reshape() ~> 이미지가 1차원 단색이라면 2차원 컬러로 변환한다.
"""
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dataset.mnist import load_mnist


if __name__ == '__main__':
    # 이미지 파일 open
    img = Image.open('sample.jpg', mode='r')

    # 이미지 객체를 numpy 배열 형태(3차원 배열)로 변환
    img_pixel = np.array(img)
    print('img_pixel shape =', img_pixel) # (height, width, color-depth)

    # MNIST 데이터 셋 load
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)

    # MNIST 데이터 shape
    print('x_train set shape =', x_train.shape)
    # x_train set shape = (60000, 1, 28, 28) ~> (samples, color, height, width)
    print('x_train[0] shape =', x_train[0].shape)
    # x_train[0] shape = (1, 28, 28) ~> (color, height, width)

    # pyplot으로 이미지 open
    plt.imshow(img_pixel)
    plt.show()

    # plt.imshow(x_train[0]) ~> (height, width, color) 순서가 아니라서 open 불가
    # 따라서 순서를 알맞게 바꿔주어야 한다. (color, height, width) ---> (height, width, color)

    num_img = np.moveaxis(x_train[0], 0, 2) # np.moveaxis(A, x, y) ~> A의 x번 축을 y번 축으로 바꾼다
    print('num_img shape =', num_img.shape) # num_img shape = (28, 28, 1) ~> (height, width, color)
    num_img = num_img.reshape((28, 28)) # axis을 변경한 이미지를 2차원 28x28 형태로 변환(단색(1차원)에서 컬러(2차원)로)

    # axis와 차원을 변경한 이미지를 다시 open
    plt.imshow(num_img)
    plt.show()