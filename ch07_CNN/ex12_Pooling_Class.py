"""
Pooling Layer Class 구현
= 4차원 데이터를 'im2col 함수'로 2차원 데이터로 변환 후, Max-Pooling 구현

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np
import matplotlib.pyplot as plt

from common.util import im2col
from dataset.mnist import load_mnist


class Pooling:
    def __init__(self, fh, fw, stride=1, pad=0):
        self.fh = fh # pooling 영역(window)의 height
        self.fw = fw # pooling 영역(window)의 width
        self.stride = stride # window 이동 간격(보폭)
        self.pad = pad # padding 값

        # backward에서 사용 할 값
        self.x = None # Pooling layer로 forward(들어오는)되는 데이터
        self.arg_max = None # Max-Pooling으로 찾은 최대값의 인덱스

    def forward(self, x):
        """
        x(n, c, h, w) ~> (samples, channel, height, width) 형태의 4차원 배열

        입력 데이터 x의 Pooling Forward Propagation
        """
        n, c, h, w = x.shape # n: 데이터 개수(samples), c: 채널 개수, h: height, w: width

        oh = int((h - self.fh) / self.stride + 1) # output height
        ow = int((w - self.fw) / self.stride + 1) # output width
        # 한편, oh / ow는 정수로 나누어 떨어져야 한다

        # 4차원 데이터 x를 2차원 ndarrya로 전개 후, 채널 별 최대 값을 쉽게 찾고자 2차원 배열로 전개된 x의 shape 변환
        col = im2col(x, self.fh, self.fw, self.stride, self.pad)
        col = col.reshape(-1, self.fh * self.fw)
        print('im2col col =', col)

        # 2차원으로 전개된 데이터에서 row 별 최대값 찾기(max pooling), 그 최대값의 인덱스 찾기
        out = np.max(col, axis=1)
        print('pooled out =', out)

        # 1차원 pooling의 결과를 4차원으로 변환(n, oh, ow, c)하고, transpose()로 축 순서를 변환(n, c, oh, ow)
        # 채널(color depth) 축이 가장 마지막 축이 되도록 reshape()하고, 채널 축이 2번째 축이 되도록 transpose()
        out = out.reshape(n, oh, ow, c).transpose(0, 3, 1, 2)
        print('pooled final out =', out)

        return out


if __name__ == '__main__':
    np.random.seed(116)

    # 입력 데이터 x (x(N, C, H, W) = (1, 3, 4, 4)의 4차원 데이터)
    x = np.random.randint(10, size=(1, 3, 4, 4))

    # 입력 데이터 x에 대한 Pooling 클래스 테스트
    pool = Pooling(fh=2, fw=2, stride=2, pad=0)  # Pooling 클래스 객체 생성
    pool.forward(x)
    # im2col col = [[5. 6. 1. 1.]
    #  [3. 5. 3. 9.]
    #  [3. 6. 3. 8.]
    #  [1. 2. 8. 7.]
    #  [9. 3. 6. 0.]
    #  [5. 7. 5. 7.]
    #  [2. 8. 3. 3.]
    #  [3. 7. 1. 5.]
    #  [3. 5. 7. 6.]
    #  [7. 3. 9. 4.]
    #  [3. 1. 3. 7.]
    #  [9. 3. 8. 4.]]

    # pooled out = [6. 9. 8. 8. 9. 7. 8. 7. 7. 9. 7. 9.]

    # pooled final out = [[[[6. 8.]
    #    [8. 9.]]
    #
    #   [[9. 9.]
    #    [7. 7.]]
    #
    #   [[8. 7.]
    #    [7. 9.]]]]

    # MNIST 데이터에 대한 Pooling 클래스 테스트 =======================

    # MNIST 데이터 load
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)

    # 학습 데이터 5개를 batch로 forward
    x = x_train[:5]
    out = pool.forward(x)
    print('x shape =', x.shape)  # x shape = (5, 1, 28, 28)
    print('out shape =', out.shape)  # out shape = (5, 1, 14, 14)

    # 학습 데이터 5개와 forwarding 된 결과에 대한 pyplot 시각화
    for i in range(5):
        ax = plt.subplot(2, 5, (i+1)) # 한 polt에 이미지를 하나씩 2행 5열로 1번 ~ 5번에 그리겠다.
        plt.imshow(x[i].squeeze(), cmap='gray') # pooling 전의 학습 데이터 그리기(squeeze(): 차원 축소)
        ax2 = plt.subplot(2, 5, (i + 6)) # 한 polt에 이미지를 하나씩 2행 5열로 6번 ~ 10번에 그리겠다.
        plt.imshow(out[i].squeeze(), cmap='gray') # pooling 후의 학습 데이터 그리기(squeeze(): 차원 축소)
    plt.show()
    # 참고) subplot에서 그래프는 0번이 아닌 1번부터 그려진다.(그래서 (i+1), (i+6)인 것)
