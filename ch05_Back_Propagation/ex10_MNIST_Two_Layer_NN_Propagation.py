"""
'오차 역전파'를 사용한 'MNIST DATA SET'의 'Two-Layer Neural Network'(hidden_layer 1개, output_layer 1개)

X -> [Affine W1, b1] -> [ReLU] -> [Affine W2, b2] -> [SoftmaxWithLoss] -> L

앞서 구현한 'Affine', 'ReLU', 'SoftmaxWithLoss' 클래스들을 사용한 신경망 구현
"""
import numpy as np

from collections import OrderedDict
from ch05_Back_Propagation.ex05_Relu import Relu
from ch05_Back_Propagation.ex07_Affine import Affine
from ch05_Back_Propagation.ex08_Softmax_Loss import SoftmaxWithLoss
from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        신경망의 구조 결정
        """
        np.random.seed(106)

        # Y = X @ W + b
        # Weight / bias 행렬 초기화
        self.params = dict() # W/b가 저장될 dict
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        # ~> OrderedDict(): 순서가 있는 dict. dict에 추가한 순서를 기억한다.
        # 그래서 'Forward Propagation'에서는 추가한 순서대로 각 layer의 forward() 메소드를 호출하고,
        # 'Back Propagation'에서는 그 반대의 순서대로 각 layer의 backward() 메소드를 호출하면 된다.
        self.layers['affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = Relu()
        self.layers['affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        input x를 받아 Forwrd Propagation을 통해 예측된 output

        Y1 = self.layer['affine1].forward(x)
        Y2 = self.layer['relu].forward(Y1)
        Y3 = self.layer['affine2].forward(Y2)

        return Y3
        """
        for layer in self.layers.values(): # OrderedDict() 타입의 layer에서의 value들을
            x = layer.forward(x) # 하나씩 forward() 메소드에 적용

        return x

    def loss(self, x, y_true):
        """
        손실함수(CEE, Cross Entropy Error)계산

        출력층(SoftmaxWithLoss)전까지의 Forward Propagation을 계산하고,
        이 결과를 출력층(SoftmaxWithLoss)을 통과시켜 손실 함수 CEE 값을 구한다.

        x: input Data
        y_true: 정답 레이블
        """
        # 출력층(SoftmaxWithLoss)전까지의 Forward Propagation을 계산
        y_pred = self.predict(x)

        # last_layer.forward() 메소드는 예측값과 실제값을 받아 SotfmaxWithLoss 함수를 통과하며 CEE를 계산한다.
        loss = self.last_layer.forward(y_pred, y_true)

        return loss

    def accuracy(self, x, y_true):
        """
        Input x의 예측 값과 실제 값의 비교를 통한 정확도 계산

        여기서 구한 '최대값의 인덱스'가 '예측한 숫자 클래스'가 된다.
        예를 들어, '최대값의 인덱스가 1'이면 '숫자 2'라고 예측한 것이고, '최대값의 인덱스가 3'이면 '숫자 4'라고 예측한 것.

        x: Input Data
        y_true: 정답 레이블
        그리고 Input과 y_true는 모두 2차원 리스트(행렬)이리고 가정한다.
        """
        y_pred = self.predict(x) # Input에 대한 예측
        y_pred = np.argmax(y_pred, axis=1) # 예측값의 각 컬럼중에서 최대값의 인덱스들을 찾음
        if y_true.ndim != 1: # 정답 레이블이 1차원이 아니면
            y_true = np.argmax(y_true, axis=1) # 정답 레이블의 각 컬럼중에서 최대값의 인덱스들을 찾는다.

        # 정확도 = (예측값 == 실제값)의 총 개수 / Input의 row 개수(전체 input data의 개수)
        accuracy = np.sum(y_pred == y_true) / float(x.shape[0])

        return accuracy

    def gradient(self, x, y_true):
        """
        Input x와 정답 레이블 y_true가 주어졌을 때,
        모든 layer에 대해 Forward propagation을 수행한 후,
        Back propagation을 통해 dW1, db1, dW2, db2를 계산

        즉, Weight / bias 행렬들의 각 값에 대한 gradient(기울기, 미분값) 계산

        x: Input Data
        y_true: 정답 레이블
        """
        # Forward Propagation
        self.loss(x, y_true)

        # Back Propagation
        # Output -> SoftmaxWithLoss
        dout = 1
        # SoftmaxWithLoss -> Affine2
        dout = self.last_layer.backward(dout)

        # 각 layer들의 value를 리스트 타입으로 구성
        # [affine1, relu, affine2]
        layers = list(self.layers.values())

        # '저장된 순서가 기억'된 'OrderedDict 타입'인 'layers'를 '역전파를 위해 반대로 뒤집는다'.
        # [affine1, relu, affine2] ~~~> [affine2, relu, affine1]
        layers.reverse()

        for layer in layers: # 순서가 반대로 뒤집인 layers들에서 layer를 하나씩 꺼내어
            dout = layer.backward(dout) # 각 layer에 대한 dout을 계산한다.

        # 결과 저장
        gradients = {} # W/b 행렬에 대한 미분값을 저장할 dict
        gradients['W1'] = self.layers['affine1'].dW
        gradients['b1'] = self.layers['affine1'].db
        gradients['W2'] = self.layers['affine2'].dW
        gradients['b2'] = self.layers['affine2'].db

        return gradients


if __name__ == '__main__':
    # MNIST Data load
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 데이터 shape
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # ~> (60000, 784) (60000, 10) (10000, 784) (10000, 10)

    # 신경망 객체 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)

    # TwoLayerNetwork() 클래스에서 구성된 W / b 행렬 형태
    for key in neural_net.params:
        print(key, ':', neural_net.params[key].shape)
        # ~> W1 : (784, 32)  b1 : (32,)  W2 : (32, 10)  b2 : (10,)

    # TwoLayerNetwork() 클래스에서 구성된 각 layer 단계
    for key in neural_net.layers:
        print(key, ':', neural_net.layers[key])
        # ~> layer는 OrderedDict() 타입으로 저장되었다.

    # TwoLayerNetwork() 클래스에서 구성된 last_layer
    print(neural_net.last_layer) # 출력층인 last_layer는 'SoftmaxWithLoss'

    # predict(예측)
    Y_pred = neural_net.predict(X_train[0]) # 이미지 1장 예측
    print('1장 Y_pred =', Y_pred)
    print('1장 Y_pred 최대값의 인덱스 =', np.argmax(Y_pred)) # 1장 Y_pred 최대값의 인덱스 = 3

    Y_pred = neural_net.predict(X_train[:3]) # 이미지 3장 예측(Mini-batch)
    print('3장 Y_pred =', Y_pred)
    print('3장 Y_pred 최대값의 인덱스 =', np.argmax(Y_pred, axis=1)) # 3장 Y_pred 최대값의 인덱스 = [3 3 3]

    # loss(손실(CEE))
    loss_1 = neural_net.loss(X_train[0], Y_train[0])
    print('1장 손실 loss =', loss_1) # 1장 손실 loss = 2.298794993348815

    loss_3 = neural_net.loss(X_train[:3], Y_train[:3])
    print('3장 손실 loss =', loss_3) # 3장 손실 loss = 2.3012479002931148

    # accuracy(정확도)
    print('3장 True Y_true =', Y_train[:3])
    print('3장 정확도 acc =', neural_net.accuracy(X_train[:3], Y_train[:3])) # 3장 정확도 acc = 0.0
    print('10장 정확도 acc =', neural_net.accuracy(X_train[:10], Y_train[:10])) # 10장 정확도 acc = 0.1

    # Weight / bias 행렬의 gradient(기울기, 미분값)
    gradients = neural_net.gradient(X_train[:3], Y_train[:3])
    for key in gradients:
        print(gradients[key].shape, end=' ')
        # (784, 32) (32,) (10,) (10,)
        # ~> W1(784, 32) / b1(32, ) / W2(10, ) / b2(10, )
    print()