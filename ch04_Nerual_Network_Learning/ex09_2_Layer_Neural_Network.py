"""
2-Layer(2층 신경망)으로 이루어진 Neural-Network(신경망)
"""
import numpy as np

from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        클래스에서 사용할 변수 초기화
        Input : 784개(28x28)
        hidden_layer_1 : 뉴런 32개
        Output_layer : 뉴런 10개
        Weight 행렬(W1, W2) / bias 행렬 : 난수로 구성

        Step 1) a1 = x(1, 784) @ W1(784, 32) + b1(1, 32)
                h1 = sigmoid_function(a1)
        Step 2) a2 = h1(1, 32) @ W2(32, 10) + b2(1, 10)
                y_pred = softmax(a2)
        """
        np.random.seed(1231)
        # Weight 초기화 / bias 초기화
        self.params = {} # W/b 행렬을 저장하는 dict
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """ Input x를 받아 Weight / bias를 적용하고 활성화 함수를 통과시킨 예측 결과 y_pred 계산 """
        # W / b
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = x.dot(W1) + b1 # Input x에 hidden_layer_1의 가중치 W1, bias b1 적용
        h1 = self.sigmoid(a1) # x가 hidden_layer_1의활성화 함수인 sigmoid()를 통과한 값

        a2 = h1.dot(W2) + b2 # 그 값에 다음 hidden_layer_2의 가중치 W2, bias b2를 적용
        y_pred = self.softmax(a2) # Output_layer의 활성화 함수인 softmax() 함수를 통과한 최종 Output

        return y_pred

    def loss(self, x, y_true):
        """ Input x에 대한 결과 y_pred와 정답 레이블 y_true의 손실 함수인 '교차 엔트로피 오차(CEE)' 계산 """
        y_pred = self.predict(x)

        cee = self.cross_entropy(y_true, y_pred)

        return cee

    def accuracy(self, x, y_true):
        """
        Input x를 사용해 예측한 결과 y_pred와 정답 레이블 y_true를 비교해 NN 모델의 예측 정확도 계산

        x, y_true는 2차원 배열이며, 실제값 y_true는 one_hot_encoding 되어있다고 가정.
        """
        y_pred = self.predict(x) # Input x를 받아 Neural_Net이 예측한 y_pred
        predictions = np.argmax(y_pred, axis=1) # 예측한 값의 y축(열)애서 가장 큰 값이 있는 인덱스를 찾음
        true_values = np.argmax(y_true, axis=1) # 정답 레이블의 y축(열)에서 가장 큰 값이 있는 인덱스를 찾음
        print('predictions =', predictions)
        print('true_values =', true_values)

        # 정확도 = 예측과 정답이 같은 개수의 평균
        accuracy = np.mean(predictions == true_values)
        return accuracy

    def gradients(self, x, y_true):
        # 가중치 행렬 W, bias 행렬 b가 주어졌을 때, 예측과 결과의 손실함수인 '교차 엔트로피 오차(CEE)'를 계산하는 함수 fn_cee
        loss_fn = lambda W: self.loss(x, y_true)

        gradients = {} # CEE를 계산하는 함수 loss_fn와 가중치 W, bias b를 전달해 편미분한 '기울기'를 저장할 dict
        for key in self.params:
            gradients[key] = self.numerical_gradient(loss_fn, self.params[key])
        # 이 반복문은 다음과 같다.
        # gradients['W1'] = self.numerical_gradient(loss_fn, self.params['W1'])
        # gradients['b1'] = self.numerical_gradient(loss_fn, self.params['b1'])
        # gradients['W2'] = self.numerical_gradient(loss_fn, self.params['W2'])
        # gradients['b2'] = self.numerical_gradient(loss_fn, self.params['b2'])

        return gradients

    def sigmoid(self, x):
        """ sigmoid() 활성화 함수 """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        softmax() 활성화 함수

        softmax = exp(x_k) / sum i to n [exp(x_i)]
        """
        if x.ndim == 1: # Input x가 1차원이면,
            max_x = np.max(x)  # input의 최대값을 찾음.
            x -= max_x # 최대값을 빼주게 되면서 오버플로우 방지
            softmax = np.exp(x) / np.sum(np.exp(x))
        else: # Input x가 2차원이라면,
            x_t = x.T # 행렬 x의 전치 행렬 x_t
            max_x = np.max(x_t) # 전치행렬 x_t의 최대값을 찾음.
            x_t -= max_x # 최대값을 빼주게 되면서 오버플로우 방지
            result = np.exp(x_t) / np.sum(np.exp(x_t), axis=0) # 이때 sum은 행렬의 x축(행)기준
            softmax = result.T # 전치 행렬 x_t로 구한 결과를 다시 전치시켜서 원래대로

        return softmax

    def cross_entropy(self, y_true, y_pred) :
        """
        손실 함수 '교차 엔트로피 오차(CEE)' 계산

        두 개의 2차원 배열에서의 CEE = -1/N * sum_i[sum_k(t__ik * logP_ik)] (t=y_true / p=y_pred / N은 행의 개수)
        """
        if y_pred.ndim == 1 :  # 예측 데이터가 1차원 리스트라면
            # 1차원 리스트를 행의 개수는 1개, 열의 개수는 원본과 같은 크기의 2차원 리스트로 변환.
            y_pred = y_pred.reshape((1, y_pred.size))
            y_true = y_true.reshape((1, y_true.size))
        # 그리고 이때 y_true는 one_hot_encoding이 되어 있다고 가정한다.

        true_values = np.argmax(y_true, axis=1) # y_true에서 '1'이 있는 컬럼 위치(인덱스)를 찾는다.
        n = y_pred.shape[0] # n = y_pred의 행(row) 수 (shape: (row, column))
        rows = np.arange(n) # rows = [0, 1, 2, 3, ...]
        # 예를 들어, y_pred[[0, 1, 2], [3, 3, 9]] (=[y_pred[0, 3], y_pred[1, 3], y_pred[2, 9])
        # ~> 최대값만 들어있는 인덱스가 0번 행에서 3번, 1번 행에서 3, 2번 행에서 9번 인덱스이고, 이 인덱스들의 값들만 계산하기 위해,
        # row 0번에서는 인덱스 3번을 찾고, row 1번에서는 인덱스 3번을 찾고, row 2번에서는 인덱스 9번을 찾겠다.

        log_p = np.log(y_pred[rows, true_values]) # 그리고 이렇게 찾은 것들로만 log 계산
        entropy = -np.sum(log_p) / n # 엔트로피는 이 log 값의 총 합을 개수 n으로 나눈 '평균'

        return entropy

    def numerical_gradient(self, fn, x):
        """
        독립 변수 n개를 갖는 함수 fn에 대한 편미분

        x = [[x11, x12, x13, ...] ,
            [x21, x22, x23, ...],
            [x31, x32, x33, ...]]
        """
        h = 1e-4 # 0.0001
        gradient = np.zeros_like(x) # gradient가 저장될 x와 같은 크기의 영행렬

        # np.nditer를 사용해보자.
        with np.nditer(x, flags=['c_index', 'multi_index'], op_flags=['readwrite']) as it :
            while not it.finished:
                i = it.multi_index
                ith_value = it[0] # 원본 데이터를 임시 변수에 저장
                it[0] = ith_value + h # 원본 값을 h만큼 증가
                fh1 = fn(x) # f(x + h)
                it[0] = ith_value - h # 원본 값을 h만큼 감소
                fh2 = fn(x) # f(x - h)
                gradient[i] = (fh1 - fh2) / (2 * h)
                it[0] = ith_value # 다음 변수에 대한 계산을 위해 가중치 행렬의 원소를 원본 값으로 복구
                it.iternext()

        # 이 과정은 기존의 gradient를 구하는 과정인 다음과 같다.
        # h = 1e-4
        # x = x.astype(np.float, copy=False)  # x는 실수 타입이 되어야 한다.('copy=False'는 원본 데이터의 타입 자체를 변화)
        # gradient = np.zeros_like(x)  # 이는 np.zeros(x.shape)와 같음(x의 행, 열 크기와 같은 영행렬)
        #
        # for i in range(x.size) :  # 점 x의 리스트 사이즈만큼(전체 원소 개수만큼)
        #     ith_val = x[i]  # i번째 value = x의 i번쨰 value
        #     x[i] = ith_val + h  # f(x+h)
        #     fh1 = fn(x)
        #     x[i] = ith_val - h  # f(x-h)
        #     fh2 = fn(x)
        #     gradient[i] = (fh1 - fh2) / (2 * h)  # 편미분 값을 계산해 각 위치에 대응하는 gradient 행렬 값 변경
        #     x[i] = ith_val  # 한 변수에 대해 미분이 끝나고, 다음 변수에 대한 미분 계산을 위해 원래의 값으로 복원

        return gradient


if __name__ == '__main__':
    # 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10) # 클래스 객체 생성(__init__()을 호출)

    # W1, W2, b1, b2 shape 확인
    print(f'W1 = {neural_net.params["W1"].shape}, b1 = {neural_net.params["b1"].shape}') # W1 = (784, 32), b1 = (32,)
    print(f'W2 = {neural_net.params["W2"].shape}, b1 = {neural_net.params["b2"].shape}') # W2 = (32, 10), b1 = (10,)

    # MNIST 데이터를 사용한 neural_net.predict() Test

    # Dataset load 후, train_set / test_set으로
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # X_train[0]을 신경망에 propagation하고, 예측 결과 확인
    mnist_y_pred_0 = neural_net.predict(X_train[0])
    print('MNIST 예측 y_pred[0] =', mnist_y_pred_0)
    print('MNIST 실제 y_true[0] =', y_train[0])


    # X_train[5]을 신경망에 propagation하고, 예측 결과 확인
    mnist_y_pred_5 = neural_net.predict(X_train[:5])
    print('MNIST 예측 y_pred[:5] =', mnist_y_pred_5)
    print('MNIST 실제 y_true[:5] =', y_train[:5])

    # 정확도
    acc_5 = neural_net.accuracy(X_train[:5], y_train[:5])
    print('정확도 acc_0 =', acc_5)
    # predictions = [9 9 9 9 9]
    # true_values = [5 0 4 1 9]
    # 정확도 acc_0 = 0.2

    # 정확도와 교차 엔트로피 오차(CEE) 계산
    acc_100 = neural_net.accuracy(X_train[:100], y_train[:100])
    print('정확도 acc_100 =', acc_100)
    cee_100 = neural_net.loss(X_train[:100], y_train[:100])
    print('CEE_100 =', cee_100)

    # gradients() 메소드 테스트 - gradient(기울기) 구하기
    gradients = neural_net.gradients(X_train[:100], y_train[:100])
    for key in gradients:
        print(key, np.sum(gradients[key]))
        # W1 -0.07166112229395338
        # b1 0.0002826510026032736
        # W2 0.00018132563006645341
        # b2 2.2498443108531774e-05

    # gradient를 찾고, 이를 이용해서 Weight/bias 행렬들을 적절하게 업데이트
    lr = 0.1 # learning_rate 적용
    for key in gradients:
        neural_net.params[key] -= lr * gradients[key]

    # Mini_batch 방법으로 100개씩 잘라서 600번 학습을 한 세트로, 반복 횟수인 epoch번 학습 시키면
    epoch = 1000
    for i in range(epoch):
        for i in range(600):
            gradients = neural_net.gradients(X_train[i*100 : (i+1)*100], y_train[i*100 : (i+1)*100])
            for key in gradients :
                neural_net.params[key] -= lr * gradients[key]







