"""
우리가 구현한 'Convolution layer'와 'Pooling layer'를 조합하여,
MNIST 데이터를 인식하는 'CNN(Convolution Neural Network)'을 구현해 보자.

구현하고자 하는 CNN의 구성(교재 P.228 그림 7-2)
    입력 데이터 -> [Convolution] -> [ReLU] -> [Pooling] -> [Affine] -> [ReLU] -> [Affine] -> [Softmax] -> 출력 데이터
    (마지막 hidden layer인 '[Affine] -> [ReLU]'는 '완전-연결(fully-connected)' 조합)

<Output의 크기 계산 공식>
입력 크기(H, W) / 필터 크기(FH, FW) / 출력 크기(OH, OW) / 패딩 P / 스트라이트 S 일 때,
    Output Height OH = (H + 2P - FH) / S + 1
    Output Width OW = (W + 2P - FW) / S + 1
단, (OH, OW)는 모두 '정수로 나누어 떨어져야'한다.
"""
import numpy as np

from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
from common.trainer import Trainer
from dataset.mnist import load_mnist


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        객체(인스턴스) 초기화

        1st hidden layer: Convolution -> ReLU -> Pooling
        2nd hidden layer: Affine -> ReLU (fully-connected, 완전-연결)
        출력 layer: Affine -> SoftmaxWithLoss

        :param input_dim: 입력 데이터(채널 수, height, width) -> mnist의 경우 (1, 28, 28
        :param conv_param: convolution layer의 하이퍼 파라미터(dict)
                            ~> filter_num(필터 수) / filter_size(필터 크기) / pad(패딩) / stride(스트라이드)
        :param hidden_size: 은닉층(Affine)의 뉴런 수 -> W 행렬의 크기
        :param output_size: 출력층의 뉴런 수 -> mnist의 경우 10
        :param weight_init_std: Weight 행렬을 난수로 초기화 시의 가중치 표준편차
        """
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        # Output size = (Input_size + 2*Pad - Filter_size) / Stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        # Pooling된 Output size = filter_size * (output_size/2) * (output_size/2) ~> int type

        # 파라미터 초기화
        # 학습에 필요한 파라미터는 1st hidden layer의 convolution layer와 나머지 두 완전연결 layer의 Weight/bias
        self.params = dict() # 파라미터(Weight/bias)가 저장될 dict
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        # 참고) np.random.randn(m, n): 평균이 0, 표준편차가 1인 가우시안 표준정규분포 난수를 m행 n열 생성

        # 구현하고자 하는 CNN 구성의 흐름에 따라 계층 생성
        # Convolution / ReLU_1 / Pooling / Affine_1 / ReLU_2 / Affine_2 / SoftmaxWithLoss
        self.layers = OrderedDict() # OrderedDict(): 순서가 있는 dict
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        입력 데이터 x를 받아 추론 수행

        '__init__()'에서 layer에 추가한 계층을 맨 앞에서부터 차례로 forward() 메소드를 호출하며 그 결과를 다음 계층으로 전달
        """
        for layer in self.layers.values(): # 순서대로 계층을 하나씩 지나며
            x = layer.forward(x) # x에 대한 결과를 계산하고 다음 계층의 입력으로

        return x

    def loss(self, x, y_true):
        """
        입력 데이터 x에 대한 예측 y_pred와 실제 y_true에 대한 loss 계산

        predict() 함수의 결과를 받아 마지막 층의 forward() 메소드 호출
        즉, 첫 계층부터 마지막 계층까지 forward 처리
        """
        y_pred= self.predict(x) # x에 대한 예측 y_pred 계산
        loss = self.last_layer.forward(y_pred, y_true) # y_pred와 y_true에 대한 loss(SoftmaxWithLoss) 계산

        return loss

    def gradient(self, x, y_true):
        """
        입력 데이터 x에 대한 예측 y_pred와 실제 y_true를 사용해 Back ProPagation으로 기울기를 계산

        파라미터의 기울기는 back propagation으로 구하며, 이 과정은 forward / back를 반복한다.
        """
        # Forward Propagation
        self.loss(x, y_true)

        # Back Propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values()) # self.layers에 들어있는 각 layer들의 값을 리스트로 생성
        layers.reverse() # 'self.layers'는 '순서가 기억된 OrderedDict'이므로 역전파를 위해 반대로 뒤집는다.
        for layer in layers:
            dout = layer.backward(dout) # dout을 받아 각 layer의 역전파 값 계산

        # 각 가중치 파라미터의 기울기 계산 결과 저장
        gradients = dict()
        gradients['W1'] = self.layers['Conv1'].dW
        gradients['b1'] = self.layers['Conv1'].db
        gradients['W2'] = self.layers['Affine1'].dW
        gradients['b2'] = self.layers['Affine1'].db
        gradients['W3'] = self.layers['Affine2'].dW
        gradients['b3'] = self.layers['Affine2'].db

        return gradients

    def accuracy(self, x, y_true, batch_size=100):
        """
        x에 대한 예측 y_pred와 실제 y_true를 비교한 정확도 계산
        """
        pass


if __name__ == '__main__':
    # MNIST 데이터 셋 load
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)

    # SimpleConvNet 객체 생성
    network = SimpleConvNet()

    # 학습 -> 테스트
    trainer = Trainer(network, x_train, y_train, x_test, y_test,
                      epochs=20, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)



