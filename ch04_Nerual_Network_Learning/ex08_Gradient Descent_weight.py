"""
weight 행렬에 Gradient Descent(경사 하강법) 적용
"""
import numpy as np

from ch03_Neural_Network.ex11_Mini_Batch import softmax
from ch04_Nerual_Network_Learning.ex03_CEE import cross_entropy
from ch04_Nerual_Network_Learning.ex05_Differential import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3) # 2x3 행렬 W(weight, 가중치)를 랜덤한 값으로 생성

    def predict(self, x):
        """
        W(가중치)를 사용한 예측 수행 메소드

        :param x: 입력 데이터
        :return: 예측한 값 y_pred
        """
        z = x.dot(self.W) # 입력 데이터 x에 가중치 W 적용
        y_pred = softmax(z) # 활성화 함수 통과 후, 예측 결과인 output y_pred.

        return y_pred

    def loss(self, x, y_true):
        """
        손실함수 계산 메소드

        :param x: 입력 데이터
        :param y_true: 정답 레이블
        :return: 손실함수의 값
        """
        y_pred = self.predict(x) # 입력 데이터 x에 대한 예측 z
        cee = cross_entropy(y_pred, y_true)
        # ~> 가중치 W를 사용한 예측 후,
        # 활성화 함수를 통과한 output y_pred와  정답 레이블 t에 대한 손실 함수인 '교차 엔트로피 오차(CEE)' 계산

        return cee

    def gradient(self, x, y_true):
        """
        손실함수의 결과가 최소가 되는 gradient(기울기)를 찾는 메소드

        :param x: 입력
        :param y_true: 출력의 실제 값(y_true) -> 정답 레이블
        :return: gradient
        """
        fn = lambda W: self.loss(x, y_true)
        # 가중치 행렬 W가 주어졌을 때, 입력 x와 예측 t에 대한 손실함수(교차 엔트로피 오차)를 계산하는 함수 fn

        return numerical_gradient(fn, self.W) # 함수 fn과 입력 x에 대한 미분 값. 즉, gradient(기울기)를 계산해 리턴


if __name__ == '__main__':
    network = SimpleNetwork() # 생성자 호출. (__init__메소드 호출)
    print('W =', network.W)
    # W = [[-0.82415727 -1.39800747  1.20783861]
    #  [ 0.62209225  3.24895652  1.17681599]]

    # 입력 값 x = [0.6, 0.9]일 때, 실제 값 y_true = [0, 0, 1]이라고 가정.
    x = np.array([0.6, 0.9])
    y_true = np.array([0.0, 0.0, 1.0])

    # 입력 데이터 x에 대한 예측 값 y_pred
    y_pred = network.predict(x)

    # 실제 값 y_true와 예측 값 y_pred 비교
    print('실제 값 y_true =', y_true)
    print('예측 값 y_pred =', y_pred)
    # 실제 값 y_true = [0. 0. 1.] ~> 2번 인덱스가 가장 큰 값.
    # 예측 값 y_pred = [0.07085565 0.53406225 0.3950821]
    # ~> 이때 예측한 결과중 가장 큰값에 해당되는 노드가 다음 layer로 전달될 때 그 노드가 선택될 확률.(또는 최종 output이 될 확률)
    # 여기서는 1번 노드가 최종 output으로 선택될 가능성이 약 53%로 가장 큰 것이다.

    # 정답 레이블 y_true와 손실함수(교차 엔트로피 오차, CEE) 계산
    cee = network.loss(x, y_true)
    print('교차 엔트로피 오차 CEE =', cee) # 교차 엔트로피 오차 CEE = 0.9286614370819835
    # 따라서 이 CEE 값이 줄어들 수 있도록, 최적의 gradient(기울기)를 찾아야 한다.

    # 가중치 행렬 W를 적용한 입력 x와 실제 값 y_true에 대한 편미분 결과. 즉, gradient의 행렬인 g1
    g1 = network.gradient(x, y_true)
    print('g1 =', g1)
    # g1 = [[ 0.04251338  0.32043727 -0.36295065]
    #  [ 0.06377007  0.48065591 -0.54442597]]
    # ~> 이는 가중치 행렬 W에서,
    # 'W11을 h만큼 늘리면 손실함수의 값은 약 0.04h만큼 증가, ..., W23을 h만큼 늘리면 손실함수의 값은 약 -0.5h만큼 감소'등을 의미.
    # 그래서 손실함수의 값을 최소화한다는 관점에서는 가중치 W의 각 항목을 이 기울기 값에 따라 늘이거나 줄여야 한다.

    # learning_rate(학습률)에 따른 각 결과 변화
    lr = 0.1
    network.W -= lr * g1
    print('W =', network.W)
    # W = [[-0.82840861 -1.4300512   1.24413368]
    #  [ 0.61571525  3.20089093  1.23125859]]
    print('CEE =', network.loss(x, y_true))
    # CEE = 0.8539188030122308
    # ~~~> 학습률의 변화에 따라 가중치 행렬 W와 손실함수의 값이 조금 변화함.(줄어들었다)

    print('예측 값 y_pred =', network.predict(x))
    # 예측 값 y_pred = [0.07055001 0.50370683 0.42574315]
    # ~~~> 마찬가지로 예측 값 또한 변화되었다.

    # ------------------------------------------------------------------------------------------------------
    # 정리하자면, 결과의 정확도를 최대한 높이기 위해서 손실 함수의 값을 최대한 줄이고자 하는 것이 목표이며,
    # 이 손실 함수의 값을 줄이기 위해서는 Weight / bias / learning rate /step등의 요소들의 값을 조금씩 바꿔가면서
    # 손실 함수의 최소값을 찾는 것이 필요하다. 그리고 이 요소들의 최적값을 찾는 일련의 과정을 '학습'이라고 한다.
    # 그리고 이 요소들에 따른 손실 함수의 변화량은 '기울기'로 알수 있다.
    # ------------------------------------------------------------------------------------------------------

    # learning rate(학습률)을 변화시켜가며 W / CEE / 예측값의 변화 관찰
    print('=============================================')
    for i in range(100):
        lr += 0.01
        g1 = network.gradient(x, y_true)
        network.W -= lr * g1
        print('\n 학습률 learning_rate =', lr)
        print('기울기 g1 =', g1)
        print('가중치 W =', network.W)
        print('교차 엔트로피 오차 CEE =', network.loss(x, y_true))
        print('예측 값 y_pred =', network.predict(x))
        #  학습률 learning_rate = 0.11
        # 기울기 g1 = [[ 0.04233     0.30222403 -0.34455403]
        #  [ 0.063495    0.45333604 -0.51683104]]
        # 가중치 W = [[-0.83306491 -1.46329584  1.28203462]
        #  [ 0.6087308   3.15102396  1.28811   ]]
        # 교차 엔트로피 오차 CEE = 0.7804135753919543
        # 예측 값 y_pred = [0.06988426 0.47189937 0.45821637]

        # ...

        #  학습률 learning_rate = 1.1000000000000008
        # 기울기 g1 = [[ 0.00213016  0.00347766 -0.00560782]
        #  [ 0.00319524  0.0052165  -0.00841174]]
        # 가중치 W = [[-1.23388323 -2.59367926  2.81323635]
        #  [ 0.00750332  1.45544883  3.58491261]]
        # 교차 엔트로피 오차 CEE = 0.009219906438438013
        # 예측 값 y_pred = [0.00349242 0.00568521 0.99082237]

        # ~~~> 점진적으로 손실 함수인 CEE의 값은 줄어들고, 예측 값 y_pred가 실제 값 y_true와 비슷해지는 것을 볼 수 있다.












