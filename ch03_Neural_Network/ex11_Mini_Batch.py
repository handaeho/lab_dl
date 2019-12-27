"""
Mini-Batch : 입력 데이터를 하나로 묶어 전달해, 처리시간을 대폭 줄일 수 있다.

신경망의 각 층에서 다차원 배열의 대응하는 차원의 원소 수가 일치한다.
예를 들어, 이미지 1장의 경우를 입력할 때는 'X(784) --> W1(784x50) --> W2(50x100) --> W3(100x10) --> Y(10)'
전체적으로 원소 784개의 1차원 배열이 입력되어, 원소 10개인 1차원 배열로 출력 되는 것이다.(원래는 28x28 2차원 배열 데이터)

이를 확장해, 이미지 여러장을 한꺼번에 입력하는 경우에
이미지 100개를 묶어 한 번에 predict() 함수에 넘기는 것으로 생각해보면 다음과 같다.
'X(100x784) --> W1(784x50) --> W2(50x100) --> W3(100, 10) --> Y(100x10)'
이때 입력 X는 X의 형상을 X(100x784)로 바꾸어서 100장의 입력 데이터를 1개의 입력 데이터로 표현한 것이다.

이와 같이 입력 X(100x784) / 출력 Y(100x10)의 형상을 갖는다. 즉, 100장 입력 데이터가 한 번에 출력된다.
가령 x[0] / y[0]에는 0번째 이미지와 그 추론 결과가, x[1] / y[1]에는 1번째 이미지와 그 추론 결과가 저장되는 것이다.
"""
import pickle
import numpy as np

from ch03_Neural_Network.ex01_Activation_Function import sigmoid_function
from dataset.mnist import load_mnist


def softmax(X):
    """
    1) X - 1차원: [x_1, x_2, ..., x_n]
    1) X - 2차원: [[x_11, x_12, ..., x_1n],
                   [x_21, x_22, ..., x_2n],
                   ...]
    """
    dimension = X.ndim

    if dimension == 1: # 1차원 배열이면,
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음.
        X = X - m  # 0 이하의 숫자로 변환 <- exp함수의 overflow를 방지하기 위해서.
        y = np.exp(X) / np.sum(np.exp(X))
    elif dimension == 2: # 2차원 배열이면,
        # m = np.max(X, axis=1).reshape((len(X), 1))
        # # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum
        Xt = X.T  # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0) # 컬럼(열)을 기준으로 X의 전치행렬 xt의 최대값을 찾는다.
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T

    return y


def forward(network, x):
    """ Forward Propagation function """
    # 가중치 행렬(Weight Matrices)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬(bias matrices)
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    z1 = sigmoid_function(x.dot(W1) + b1) # hidden_layer_1 통과의 결과를 전파(propagaton)
    z2 = sigmoid_function(z1.dot(W2) + b2) # hidden_layer_2 통과의 결과를 전파(propagaton)
    y = softmax(z2.dot(W3) + b3) # output_layer 통과의 결과를 전파(propagaton)

    return y # 최종 output y를 리턴


def mini_batch(network, X, batch_size):
    """
    mini-batch를 통한 예측
    입력 데이터 X를 batch_size만큼씩 나누어 예측
    """
    y_pred = [] # 예측 값을 저장할 리스트

    # batch_size만큼씩 X를 나누어 forward propagation.
    for i in range(0, len(X), batch_size): # 입력 데이터 X를 batch_size만큼씩 step
        X_batch = X[i:i+batch_size] # X를 batch_size만큼씩 묶어서 꺼냄
        y_hat = forward(network, X_batch) # batch_size만큼씩 묶인 X를 network(W/b)와 함께 전파
        predictions = np.argmax(y_hat, axis=1) # 각 row에서 최대값의 인덱스를 찾음 -> (batch_size, ) 배열의 형태를 가짐
        y_pred = np.append(y_pred, predictions)

    return y_pred # (lex(x), )의 형태를 갖는 배열


def accuracy(y_test, y_pred):
    """ 실제값(y_test)과 예측값(y_pred)을 받아 비교해 정확도 계산"""
    
    return np.mean(y_test == y_pred)


if __name__ == '__main__':
    np.random.seed(2020)

    # 1차원 softmax 테스트
    a = np.random.randint(10, size=5)
    print(a) # [0 8 3 6 3]
    print(softmax(a)) # [2.91923255e-04 8.70210959e-01 5.86343532e-03 1.17770247e-01 5.86343532e-03]

    # 2차원 softmax 테스트
    A = np.random.randint(10, size=(2, 3))
    print(A) # [[3 7 8] [0 0 8]]
    print(softmax(A))
    # [[4.90168905e-03 2.67623154e-01 7.27475157e-01]
    #  [3.35237708e-04 3.35237708e-04 9.99329525e-01]]

    # (Train/Test) 데이터 세트 로드 ------------------------------------------------------
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print('X_test shape =', X_test.shape) # X_test shape = (10000, 784)
    print('y_test shape =', y_test.shape) # y_test shape = (10000,)

    # 신경망 생성 (W1, b1, ...) ~> 저자가 구성한 'sample_weight.pkl' 데이터를 읽어와 사용하자
    with open('sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)
    print('network =', network.keys()) # network = dict_keys(['b2', 'W1', 'b1', 'W2', 'W3', 'b3'])
    print('W1 =', network['W1'].shape) # W1 = (784, 50)
    print('W1 =', network['W2'].shape) # W2 = (50, 100)
    print('W1 =', network['W3'].shape) # W3 = (100, 10) ~~~> 각 hidden_layer의 Weigth에서, 대응하는 차원의 원소 수 일치

    # Mini-Batch를 통한 예측
    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print('실제값 y_test[:10] =', y_test[:10]) # 실제값 y_test[:10] = [7 2 1 0 4 1 4 9 5 9]
    print('예측값 y_pred[:10] =', y_pred[:10]) # 예측값 y_pred[:10] = [7. 2. 1. 0. 4. 1. 4. 9. 6. 9.]
    print('실제값 y_pred[-10:] =', y_test[-10:]) # 실제값 y_pred[-10:] = [7 8 9 0 1 2 3 4 5 6]
    print('예측값 y_pred[-10:] =', y_pred[-10:]) # 예측값 y_pred[-10:] = [7. 8. 9. 0. 1. 2. 3. 4. 5. 6.]

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print(acc) # 0.9352

