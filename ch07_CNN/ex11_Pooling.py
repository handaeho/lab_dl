"""
입력 데이터 x의 Pooling

단, Pooling의 경우, '채널 쪽이 독립적'이라는 점이 Convolution layer와 다르다.
pooling 적용 영역을 채널마다 독립적으로 계산하는 것.

입력 데이터 -> 전개 -> max-pooling -> reshape -> 출력 데이터
(입력 데이터를 전개한 후, 전개한 행렬에서 행별 최대값을 구하고 적절한 형상으로 성형)
"""
import numpy as np
from common.util import im2col


if __name__ == '__main__':
    np.random.seed(116)

    # 가상의 이미지 데이터 x 1개를 난수로 생성 (x(N, C, H, W) = (1, 3, 4, 4)의 4차원 데이터)
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print('X =', x)
    # X = [[[[5 6 1 2]
    #    [1 1 8 7]
    #    [2 8 7 3]
    #    [3 3 9 4]]
    #
    #   [[3 5 9 3]
    #    [3 9 6 0]
    #    [3 7 3 1]
    #    [1 5 3 7]]
    #
    #   [[3 6 5 7]
    #    [3 8 5 7]
    #    [3 5 9 3]
    #    [7 6 8 4]]]]

    print('x.shape =', x.shape)
    # x.shape = (1, 3, 4, 4)

    # im2col()로 4차원 x를 2차원 ndarray로 전개
    col = im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)
    print('im2col col =', col)
    # im2col col = [[5. 6. 1. 1. 3. 5. 3. 9. 3. 6. 3. 8.]
    #  [1. 2. 8. 7. 9. 3. 6. 0. 5. 7. 5. 7.]
    #  [2. 8. 3. 3. 3. 7. 1. 5. 3. 5. 7. 6.]
    #  [7. 3. 9. 4. 3. 1. 3. 7. 9. 3. 8. 4.]]

    print('im2col col shape =', col.shape)
    # im2col col shape = (4, 12) ~> (n*oh*ow, c*fh*fw) ~> (window 이동 횟수, 모든 채널에서의 필터 원소 수)

    # max pooling: 채널 별로 '최대 값'을 찾음
    # 채널 별 최대 값을 쉽게 찾고자 2차원 배열로 전개된 x의 shape 변환
    # ~> 변환된 행렬의 각 행에는, 채널 별로 window에 포함된 값들로만 이루어진다.
    col = col.reshape(-1, 2 * 2) # ~> (-1, fh * fw)
    # reshape()에서 '-1'을 지정하면 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 적절히 묶어준다.
    print('col reshape=', col)
    # col reshape= [[5. 6. 1. 1.]
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

    print('col reshape shape =', col.shape)
    # col reshape shape = (12, 4) ~> window 사이즈만큼의 1개의 채널이 각각 1개의 배열로(1번 영역의 1번채널, 1번 영역의 2번채널, ...)

    # 이렇게 변환된 배열의 각 행(row)에서 최대 값을 찾는다.
    out = np.max(col, axis=1) # axis=1 ~> 행(row) 방향 기준
    print('Max out =', out)
    # Max out = [6. 9. 8. 8. 9. 7. 8. 7. 7. 9. 7. 9.]

    print('Max out shape =', out.shape)
    # Max out shape = (12,) ~> Pooling의 결과가 1차원으로

    # 1차원 pooling의 결과를 4차원으로 변환(n, oh, ow, c)하고, transpose()로 축 순서를 변환(n, x, oh, ow)
    # 채널(color depth) 축이 가장 마지막 축이 되도록 reshape()하고
    out = out.reshape(1, 2, 2, 3)
    print('out =', out)
    # out = [[[[6. 9. 8.]
    #    [8. 9. 7.]]
    #
    #   [[8. 7. 7.]
    #    [9. 7. 9.]]]]

    print('out shape =', out.shape)
    # out shape = (1, 2, 2, 3)

    # 채널 축이 2번째 축이 되도록 transpose()
    out = out.transpose(0, 3, 1, 2)
    print('out =', out)
    # out = [[[[6. 8.]
    #    [8. 9.]]
    #
    #   [[9. 9.]
    #    [7. 7.]]
    #
    #   [[8. 7.]
    #    [7. 9.]]]]

    print('out shape =', out.shape)
    # out shape = (1, 3, 2, 2)
