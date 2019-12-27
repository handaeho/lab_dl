"""
ex04에 이어 같은 구조의 Neural Network를 구성해 보자.

어떤 신경망이 input x의 output y에 대하여 2개의 hidden layer를 갖는 3층 구조라고 할 때,
input x(x1, x2) -> 1층 hidden_layer_1 (3개 노드) -> 2층 hidden_layer_2 (2개 노드) -> output y(y1, y2)
한 층의 출력은 다음 층의 입력이 된다. 그리고 각 hidden_layer에서는 'activation function'이 동작한다.
"""
import numpy as np

from ch03_Neural_Network.ex01_Activation_Function import sigmoid_function


def init_network():
    """
    Neural Network에서 사용되는 Weight 행렬과 bias 행렬
    - input_layer : x(x1, x2) ~> 1x2 행렬
    - hidden_layer 2개
        -- hidden_layer_1 = 뉴런 3개 (h1 = x @ W1 + b1)
        -- hidden_layer_2 = 뉴런 2개 (h2 = h1 @ W2 + b2)
    - output_layer : y(y1, y2) ~> 1x2 행렬 (y = h2 @ W3 + b3)

    단, W1, W2, W3, b1, b2, b3은 난수로 생성
    """
    np.random.seed(1224)
    network = {} # W/b 행렬을 저장하기 위한 dict. 최종 return 값

    # STEP 1) h1 = x @ W1 + b1 ~> 다음 layer(h2)에 전달하기 위하여 1x3 행렬이 되어야 한다.
    # ---> x(1, 2) @ W1(2, 3) + b(1, 3)
    network['W1'] = np.random.random(size=(2, 3)).round(2)
    network['b1'] = np.random.random(size=(1, 3)).round(2)

    # STEP 2) h2 = h1 @ W2 + b2 ~> 다음 layer(y)에 전달하기 위하여 1x2 행렬이 되어야 한다.
    # ---> h2(1, 2) = h1(1, 3) @ W2(3, 2) + b(1, 2)
    network['W2'] = np.random.random(size=(3, 2)).round(2)
    network['b2'] = np.random.random(size=(1, 2)).round(2)

    # STEP 3) y = h2 @ W3 + b3 ~> 최종 output을 위해 1x2 행렬이 되어야 한다.
    # ---> y(1, 2) = h2(1, 2) @ W3(2, 2) + b(1, 2)
    network['W3'] = np.random.random(size=(2, 2)).round(2)
    network['b3'] = np.random.random(size=(1, 2)).round(2)

    return network # 구성된 각 단계의 Weigth 행렬 / bias 행렬을 return한다.


def forward(network, x):
    """
    :param network: 신경망에서 사용되는 W/b 행렬들을 저장한 dict
    :param x: input(1차원 리스트 [x1, x2])
    :return: 2개의 hidden_layer와 output_layer를 거쳐 계산된 최종 output

    구성된 NN의 '각 layer의 결과 a'와 'Sigmoid 활성화 함수를 적용한 output z' 그리고 '최종 output y'를 계산
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1) + b1
    z1 = sigmoid_function(a1)

    a2 = z1.dot(W2) + b2
    z2 = sigmoid_function(a2)

    a3 = z2.dot(W3) + b3
    y = a3

    # output_layer의 활성화 함수는 softmax 함수
    return softmax(y)

def identity_function(x) :
    """
    identity_function(항등 함수) : input 그대로 return

    일반적으로 '회귀(연속적인 값) 문제'에 사용
    """
    return x


def softmax(x):
    """
    softmax : input이 softmax 함수를 거쳐 output_1, output_2, ... 등으로 '분류'된다.
    ~> f(x_k) = exp(x_k) / ∑(exp(x_k))

    일반적으로 '분류 문제'에 사용(즉, '확률'의 개념)

    softmax 함수의 결과는 0 ~ 1 사이의 값이 되며, 모든 결과 값의 총 합은 1이다.
    이런 특징 때문에, softmax 함수의 출력 값은 확률로 해석될 수 있다.

    그러나, softmax 함수는 지수함수(exp)를 사용하기 때문에, 값이 급격이 커져 '오버플로우'의 문제가 존재한다.
    따라서 '분모와 분자에 정수 C'를 취하고, 'exp() 안으로 옮기며 logC'가 되게 하고,
    '이 값을 빼주면서' 오버플로우 발생을 방지한다.
    ~> f(x_k) = exp(x_k) / ∑(exp(x_k)) = exp(x_k ± logC) / ∑(exp(x_k) ± logC)

    여기서는 'input의 최대값을 빼주게 되는 것'. (오버플로우 방지니까)
    """
    c = np.max(x) # c = logC. 즉, input의 최대값을 찾음.
    exp_x = np.exp(x - c) # input에서 최대값 c를 빼주어 오버플로우를 방지.
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


if __name__ == '__main__':
    network = init_network()
    x = np.array([1, 2])
    y = forward(network, x)
    print(f'input {x}에 대한 NN의 output {y}')

    # softmax() 테스트
    print('x =', x)
    print('softmax(x) =', softmax(x))

    x = [1, 2, 3]
    print('x =', x)
    print('softmax(x) =', softmax(x))


