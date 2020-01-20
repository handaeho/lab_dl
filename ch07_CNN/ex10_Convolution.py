"""
Convolution Layer Class 구현

im2col() 함수(image to column, 이미지에서 행렬로)
= 입력 데이터를 필터링(가중치 계산)하기 편하게 '펼치는' 함수.
3차원 입력 데이터에 'im2col 함수'를 적용하면 2차원 행렬로 바뀐다.
(정확하게는 batch 안의 데이터 수까지 포함한 4차원 데이터를 2차원으로 변환하는 것)

(참고)
예제가 아닌 실제 상황에서는 필터의 적용 영역에 대한 스트라이드의 간격이 겹치는 경우가 대부분이다.
필터의 적용 영역이 겹치게 되면 im2col 함수로 전개한 후의 원소 수가 원래 블록의 원소 수보다 많아지게 된다.
그래서 im2col 함수를 사용해 구현하면 메모리를 더 많이 소비하는 단점이 있다.

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import matplotlib.pyplot as plt
import numpy as np
from common.util import im2col
from dataset.mnist import load_mnist


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # weight - filter
        self.b = b  # bias
        self.stride = stride
        self.pad = pad
        # 중간 데이터: forward에서 생성되는 데이터 -> backward에서 사용
        self.x = None
        self.x_col = None
        self.W_col = None
        # gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        """x: 4차원 이미지 (mini-batch) 데이터"""
        self.x = x
        n, c, h, w = self.x.shape
        fn, c, fh, fw = self.W.shape
        oh = (h - fh + 2 * self.pad) // self.stride + 1  # output height
        ow = (w - fw + 2 * self.pad) // self.stride + 1  # output width

        self.x_col = im2col(self.x, fh, fw, self.stride, self.pad)
        self.W_col = self.W.reshape(fn, -1).T
        # W(fn,c,fh,fw) --> W_col(fn, c*fh*fw) --> (c*fh*fw, fn)
        # 입력 데이터를 im2col()로 전개하고 필터도 reshape()를 사용해 2차원 배열로 전개한다.
        # reshape()에서 '-1'을 지정하면 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 적절히 묶어준다.

        out = np.dot(self.x_col, self.W_col) + self.b
        # self.x_col.dot(self.W_col)
        # 2차원 배열로 전개된 두 행렬의 dot 연산 수행

        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        # 출력 데이터를 적절한 형상으로 바꿔준다.
        # transpose(): 다차원 배열의 축 순서를 변환.(N(0), H(1), W(2) ,C(3)) ~~~> (N(0), C(3), H(1), W(2))

        return out


if __name__ == '__main__':
    np.random.seed(115)

    # Convolution을 생성
    # filter: (fn, c, fh, fw) = (1, 1, 4, 4)
    W = np.zeros((1, 1, 4, 4), dtype=np.uint8)  # dtype: 8bit 부호없는 정수
    W[0, 0, 1, 1] = 1
    b = np.zeros(1)
    conv = Convolution(W, b)  # Convolution 클래스의 생성자 호출

    # MNIST 데이터를 forward
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False,
                                                      flatten=False)
    input = x_train[0:1]
    print('input:', input.shape)
    out = conv.forward(input)
    print('out:', out.shape)

    img = out.squeeze()
    print('img:', img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()

    # 다운로드 받은 이미지 파일을 ndarray로 변환해서 forward











