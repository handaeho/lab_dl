"""
6.2 Weight Init(가중치 초기값)

Y = X @ W + b
신경망의 파라미터인 Weight(가중치) 행렬을 처음에 어떻게 초기화를 하는가에 따라 신경망의 성능이 달라질 수 있다.

Weight의 초기값을 모두 0으로 하거나 모두 같은 값으로 설정하면, 학습이 이루어 지지 않음.
그래서 'Weight의 초기값'은 '정규분포를 따르는 난수를 랜덤하게 추출'해서 만든다.
그러나 이때 정규분포의 표준편차에 따라 학습 성능이 달라지기도 한다.

1) Weight 행렬의 초기값을 평균이 0, 표준편차가 1인 난수로 생성하면, 활성화 값들이 0 또는 1에 치우쳐서 분포한다.
   그래서 역전파의 gradient들이 점점 작아지다가 사라지게된다.
   = '기울기 소실(gradient vanishing)' 문제 발생

2) Weight 행렬의 초기값을 평균이 0, 표준편차가 0.01인 난수로 생성하면, 활성화 값들이 0.5 부근에 몰려서 분포한다.
   그래서 다수의 뉴런이 거의 같은 값을 출력하게 되므로 뉴런을 여러개 구성한 의미가 없어진다.
   = '표현력 약화' 문제 발생

3) 'Xavier 초기값'
    ~> 위와 같은 문제들을 해결하기 위해, Weight의 초기값의 표준편차를 앞 계층의 노드 숫자를 고려해 생성한다.
    'Xavier 초기값'에서의 '표준편차'는 '1/sqrt(n)'이고, 'n'은 '앞 층의 노드 수'이다.
    앞 층에 노드가 많을수록 대상 노드의 초기값으로 설정하는 가중치가 좁게 퍼진다.

그리고 'Sigmoid 함수'는 '(x, y) = (0, 0.5)'에서 대칭인 곡선이지만, 'tanh 함수'는 '원점'에서 대칭인 곡선이다.
일반적으로 활성화 함수는 '원점에서 대칭'인 함수가 바람직하며, 'tanh 함수'를 사용하면 가중치 분포의 일그러짐을 개선할 수 있다.

4) 'He 초기값'
   ~> 'Xavier 초기값'은 활성화 함수가 '선형'임을 전제로 한다. 그래서 'Relu 함수'를 이용할 때는 올바르지 않다.
   따라서 'ReLU 함수'를 활성화 함수로 사용할 때는 'He 초기값'을 이용한다.
   'He 초기값'에서의 활성화 값 표준 편차는 'sqrt(2/n)'을 따른다.
   'Relu함수'은 음의 영역이 0이기 때문에 'Xavier 초기값' 표준편차의 2배 계수를 사용하는 것이다.
"""
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


if __name__ == '__main__':
    # 은닉층(hidden layer)에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-5, 5, 100)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.title('Activation Functions')
    plt.ylim((-1.5, 1.5))
    plt.axvline(color='0.9')
    plt.axhline(color='0.9')
    plt.axhline(1, color='0.9')
    plt.axhline(-1, color='0.9')
    plt.plot(x, y_sig, label='Sigmoid')
    plt.plot(x, y_tanh, label='Hyperbolic tangent')
    plt.plot(x, y_relu, label='ReLU')
    plt.legend()
    plt.show()

    # 가상의 신경망에서 사용할 테스트 데이터(mini-batch)를 생성
    np.random.seed(108)
    x = np.random.randn(1000, 100)  # 정규화가 된 테스트 데이터

    node_num = 100  # 은닉층의 노드(뉴런) 개수
    hidden_layer_size = 5  # 은닉층의 개수
    activations = dict()  # 데이터가 은닉층을 지났을 때 출력되는 값을 저장

    weight_init_types = {
        'std=0.01': 0.01,
        'Xavier': np.sqrt(1/node_num),
        'He': np.sqrt(2/node_num)
    }
    input_data = np.random.randn(1_000, 100)
    for k, v in weight_init_types.items():
        x = input_data
        # 입력 데이터 x를 5개의 은닉층을 통과시킴.
        for i in range(hidden_layer_size):
            # 은닉층에서 사용하는 가중치 행렬:
            # 평균 0, 표준편차 1인 정규분포(N(0, 1))를 따르는 난수로 가중치 행렬 생성
            # 1) 표준편차 1: w = np.random.randn(node_num, node_num)
            # 2) 표준편차 0.01: w = np.random.randn(node_num, node_num) * 0.01  # N(0, 0.01)
            # 3) 'Xavier 초기값': w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num)  # N(0, sqrt(1/n))
            # 4) 'He 초기값': w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)  # N(0, sqrt(2/n))
            w = np.random.randn(node_num, node_num) * v
            a = x.dot(w)  # a = x @ w
            # x = sigmoid(a)  # 활성화 함수 적용 -> 은닉층의 출력(output)
            x = tanh(a)
            # x = relu(a)
            activations[i] = x  # 그래프 그리기 위해서 출력 결과를 저장

        for i, output in activations.items():
            plt.subplot(1, len(activations), i+1)
            # subplot(nrows, ncols, index). 인덱스는 양수(index >= 0).
            plt.title(f'{i+1} layer')
            plt.hist(output.flatten(), bins=30, range=(-1, 1))
        plt.show()
