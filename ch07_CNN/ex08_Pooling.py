"""
Pooling(풀링)
= 세로/가로의 공간을 줄이는 연산. 일정 크기의 영역을 원소 하나로 집약하여 공간 크기를 줄인다.

- Max Pooling(최대 풀링) ~> 대상 영역 크기 안에서 '최대값'을 구해 그 영역의 원소 값으로 삼는다.
- Average Pooling(평균 풀링) ~> 대상 영역의 크기 안에서 원소들의 '평균값'을 구해, 그 영역의 원소 값으로 삼는다.

이미지 인식 분야에서는 일반적으로 'Max Pooling(최대 풀링)'을 사용한다.

<장점>
    - 학습해야 할 파라미터(매개변수)가 없다.
    - 채널 수가 변하지 않는다. 입력 데이터의 채널 수와 동일하게 출력으로 내보낸다.(채널마다 독립적으로 계산하기 때문)
    - 입력 데이터의 변화에 영향을 적게 받는다.

참고로 Poolling-Size(풀링의 윈도우 크기)와 Stride(스트라이드, 이동 간격)는 같은 값(같은 크기)으로 설정하는 것이 일반적이다.

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from PIL import Image


def pooling1d(x, pool_size, stride=1):
    """
    1차원 입력 데이터 x의 Pooling

    :param x: 1차원 ndarray
    :param pool_size: pooling window size
    :param stride: window가 이동할 간격(보폭)
    :return: max-pooling
    """
    n = x.shape[0] # len(x)
    result_size = (n - pool_size) // stride + 1
    result = np.zeros(result_size) # 결과가 저장될 영행렬
    for i in range(result_size):
        x_sub = x[i*stride:(i*stride)+pool_size] # pool-size 영역을 스트라이드 간격만큼 이동하며
        result[i] = np.max(x_sub) # 최대값 찾아 그 영역의 대표 원소 값으로

    return result


def pooling2d(x, pool_h, pool_w, stride=1):
    """
    2차원 입력데이터 x의 Pooling

    :param x: 2차원 ndarray
    :param pool_h: pooling window height
    :param pool_w: pooling window width
    :param stride: window가 이동할 간격(보폭)
    :return: max-pooling
    """
    h, w = x.shape[0], x.shape[1] # 입력데이터 x의 height, width
    oh = (h - pool_h) // stride + 1 # 출력 데이터 output height(oh)
    ow = (w - pool_w) // stride + 1 # 출력 데이터 output width(ow)
    result = np.zeros((oh, ow)) # pooling 값이 저장될 영행렬
    for i in range(oh):
        for j in range(ow):
            x_sub = x[i*stride:(i*stride)+pool_h, j*stride:(j*stride)+pool_w]
            # (pool_h, pool_w) 크기 영역을 stride 간격만큼 이동하며
            result[i][j] = np.max(x_sub)
            # 그 영역의 최대값을 찾고 result에 저장

    return result


if __name__ == '__main__':
    np.random.seed(114)

    # 1차원 데이터 Pooling
    x = np.random.randint(10, size=10)
    print(x) # [3 6 2 2 8 9 3 8 4 4]

    pooled_1dim = pooling1d(x, pool_size=2, stride=2)
    print(pooled_1dim) # [6. 2. 9. 8. 4.]

    pooled_1dim_2 = pooling1d(x, pool_size=4, stride=2)
    print(pooled_1dim_2) # [6. 9. 9. 8.] ~> pool-size가 4인데, stride가 2여서 겹치는 영역 생김

    pooled_1dim_3 = pooling1d(x, pool_size=3, stride=3)
    print(pooled_1dim_3) # [6. 9. 8.]

    # 2차원 데이터 Pooling
    x = np.random.randint(100, size=(8, 8))
    print(x)

    pooled_2dim = pooling2d(x, pool_h=2, pool_w=2, stride=2)
    print(pooled_2dim)

    x = np.random.randint(100, size=(5, 5))
    print(x)

    pooled_2dim_2 = pooling2d(x, pool_h=3, pool_w=3, stride=2)
    print(pooled_2dim_2)

    # ==========================================

    # MNIST 데이터 셋 load
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)

    # 이미지 랜덤하게 선택
    img = x_train[0]

    # 선택한 이미지 shape(1, 28, 28)을 (28, 28)로 변환
    img_2d = img.reshape((28, 28)) # img_2d = img[0:, :, :]과 같다.

    # 이미지를 pyplot로 출력
    plt.imshow(img_2d)
    plt.show()

    # window shape = (4, 4), stride = 4로 pooling ~> output shape = (7, 7)
    img_pooled = pooling2d(img_2d, pool_h=4, pool_w=4, stride=4)
    print('img_pooled output shape =', img_pooled.shape) # img_pooled output shape = (7, 7)

    # output을 pyplot으로 출력해, pooling 전과 비교
    plt.imshow(img_pooled)
    plt.show()
    # ~> Pooling을 통해 공간을 축소함으로써, 원본보다 훨씬 뭉개져서 특징만 간략하게 표현되었음을 볼 수 있다.

    # ==========================================

    # sample.jpg 파일 open
    sample_img = Image.open('sample.jpg', mode='r')

    # sampel_img를 array 타입으로
    sample_img_pixel = np.array(sample_img)
    print('sample_img_pixel.shape =', sample_img_pixel.shape) # sample_img_pixel.shape = (937, 1920, 3)
    # ~> sample_img.shape = (세로길이(height), 가로길이(width), color-depth(RGB))

    # Red, Green, Blue에 해당하는 2차원 배열들을 추출
    img_r = sample_img_pixel[:, :, 0] # Red panel
    img_g = sample_img_pixel[:, :, 1] # Green panel
    img_b = sample_img_pixel[:, :, 2] # Blue panel

    # 각각의 RGB 2차원 배열을 window shape = (16, 16), stride = 16으로 pooling
    img_r_pooled = pooling2d(img_r, pool_h=16, pool_w=16, stride=16)
    img_g_pooled = pooling2d(img_g, pool_h=16, pool_w=16, stride=16)
    img_b_pooled = pooling2d(img_b, pool_h=16, pool_w=16, stride=16)

    # pooling된 결과의 shape를 확인하고 pyplot으로 출력
    print('Pooled Red image Panel Shape =', img_r_pooled.shape) # Pooled Red image Panel Shape = (58, 120)
    print('Pooled Green image Panel Shape =', img_g_pooled.shape) # Pooled Green image Panel Shape = (58, 120)
    print('Pooled Blue image Panel Shape =', img_b_pooled.shape) # Pooled Blue image Panel Shape = (58, 120)

    img_all_pooled = np.array([img_r_pooled, img_g_pooled, img_b_pooled]).astype(np.uint8)
    print('image all pooled shape =', img_all_pooled.shape)
    # image all poolied shape = (3, 58, 120) ~> (channel, height, width)
    # 따라서 (height, width, channel)로 바꿔 주어야 출력이 가능하다.
    move_axis_img_all_pooled = np.moveaxis(img_all_pooled, 0, 2)
    print('move axis image all pooled shape =', move_axis_img_all_pooled.shape)
    # move axis image all pooled shape = (58, 120, 3) ~> (height, width, channel)

    plt.imshow(img_all_pooled)
    plt.show()