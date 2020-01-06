"""
X -> [Affine W1, b1] -> [ReLU] -> [Affine W2, b2] -> [SoftmaxWithLoss] -> L

앞서 구현한 'Affine', 'ReLU', 'SoftmaxWithLoss' 클래스들을 사용한 신경망 구현
"""
import numpy as np

from ch05_Back_Propagation.ex05_Relu import Relu
from ch05_Back_Propagation.ex07_Affine import Affine
from ch05_Back_Propagation.ex08_Softmax_Loss import SoftmaxWithLoss

np.random.seed(106)

# Input Data: (1, 2) shape의 np.ndarray
X = np.random.rand(2).reshape((1, 2))
print('X =', X)

# True Label
Y_true = np.array([1, 0, 0])
print('Y_true =', Y_true)

# 1st_hidden_layer
# 1st_hidden_layer에서 사용할 Weight / bias 행렬
# 1st_hidden_layer의 뉴런 개수 = 3개
# W1 shape: (2, 3) / b1 shape: (3, ) ~> X(1, 2)와의 dot 연산과 덧셈 연산을 위해 이런 형태여야 한다.
W1 = np.random.randn(2, 3)
b1 = np.random.rand(3)
print('W1 =', W1)
print('b1 =', b1)

# 1st_affine_layer
affine1 = Affine(W1, b1)

# 1st_Activation_function_layer ~> ReLU()
relu = Relu()

# Output_layer
# Output_layer의 뉴런 개수 = 3개
# W2 shape: (3, 3) / b2 shape : (3, ) ~> 전 단계의 출력이 입력이 되고, 그것과의 dot 연산과 덧셈 연산을 위해 이런 형태여야 한다.
W2 = np.random.randn(3, 3)
b2 = np.random.rand(3)
print('W2 =', W2)
print('b2 =', b2)

# 2nd_affine_layer
affine2 = Affine(W2, b2)

# output_Activation_function_layer ~> SoftmaxWithLoss()
last_layer = SoftmaxWithLoss()

# 구성된 각 layer들을 연결 ~> 순전파(Forward Propagation)
# X -> [Affine W1, b1] -> [ReLU] -> [Affine W2, b2] -> [SoftmaxWithLoss] -> L
Y = affine1.forward(X)
print('Y shape =', Y.shape) # Y shape = (1, 3)

Y = relu.forward(Y)
print('Y shape =', Y.shape) # Y shape = (1, 3)

Y = affine2.forward(Y)
print('Y shape =', Y.shape) # Y shape = (1, 3)

loss = last_layer.forward(Y, Y_true)
print('loss =', loss) # loss = 1.488383731382226 ~> Cross Entropy Error

# 예측값 Y_pred와 실제값 Y_true 비교
Y_pred = last_layer.y_pred
print('Y_true =', Y_true) # Y_true = [1 0 0]
print('Y_pred =', Y_pred) # Y_pred = [[0.22573711 0.2607098  0.51355308]]

# Gradient를 계산하기 위한 역전파(Back Propagation)
learning_rate = 0.1 # 학습률

# 출력 Y -> SoftmaxWithLoss
dout =last_layer.backward(1)
print('dout_1 =', dout) # dout_1 = [[-0.77426289  0.2607098   0.51355308]]

# SoftmaxWithLoss -> 2nd_affine_layer
dout = affine2.backward(dout)
print('dout_2 =', dout) # dout_2 = [[ 0.11461604  1.21444366 -1.09939752]]
print('dW2 =', affine2.dW) # W2 방향의 gradient
print('db2 =', affine2.db) # b2 방향의 gradient

# 2nd_affine_layer -> ReLU_layer
dout = relu.backward(dout)
print('dout_3 =', dout) # dout_3 = [[0.11461604  0.     0.    ]]

# ReLU_layer -> 1st_affine_layer
dout = affine1.backward(dout)
print('dout_4 =', dout) # dout_4 = [[0.24813628 0.07624834]]
print('dW1 =', affine1.dW) # W1 방향의 gradient
print('db1 =', affine1.db) # b1 방향의 gradient

# 계산한 gradient와 학습률을 이용해 Weight / bias 행렬 수정
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db

# 수정된 Weight / bias 행렬들로 다시 Forward Propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss) # loss = 1.217319617205198

# 수정된 W/b 행렬을 적용한 예측값 Y_pred와 실제값 Y_true 비교
Y_pred = last_layer.y_pred
print('Y_true =', Y_true) # Y_true = [1 0 0]
print('Y_pred =', Y_pred) # Y_pred = [[0.29602246 0.25014373 0.45383381]]
# ~> 0번 인덱스의 값이 조금 커지고, 2번 인덱스의 값이 조금 작아졌다.

# Mini-Batch를 적용한 방법은? (단, 신경망의 구성은 위의 구성과 같다) -------------------------------------------------
# Mini-Batch 방법을 사용해, 1차원 데이터를 여러개 묶어서 구성한 2차원 Input 데이터에 대한 Propagation.
# Input X
X = np.random.rand(3, 2)
print('X =', X)

# True label Y_true
Y_true = np.identity(3)
print('Y_true =', Y_true) # Y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# 참고로 Weight / bias는 Input의 '컬럼 수'에만 영향을 받기 때문에 위에서 사용한 W/b 행렬의 모양을 그대로 적용해도 된다.
affine1 = Affine(W1, b1)

relu = Relu()

affine2 = Affine(W2, b2)

last_layer = SoftmaxWithLoss()

# Forward Propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss) # loss = 1.2004980139334998

# gradient를 구하기 위한 Back Propagation
dout = last_layer.backward(1)
print('dout_1 =', dout)

dout = affine2.backward(dout)
print('dout_2 =', dout)
print('dW2 =', affine2.dW) # W2 방향의 gradient
print('db2 =', affine2.db) # b2 방향의 gradient

dout = relu.backward(dout)
print('dout_3 =', dout)

dout = affine1.backward(dout)
print('dout_4 =', dout)
print('dW1 =', affine2.dW) # W1 방향의 gradient
print('db1 =', affine2.db) # b1 방향의 gradient

# 계산한 gradient와 학습률을 이용해 Weight / bias 행렬 수정
learning_rate = 0.1

W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db

# gradient에 따라 수정된 W/b 행렬을 적용한 Forward Propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss) # loss = 1.1640299948689463 ~> 약 0.05 감소

# 수정된 W/b 행렬을 적용한 예측값 Y_pred와 실제값 Y_true 비교
Y_pred = last_layer.y_pred
print('Y_true =', Y_true)
print('Y_pred =', Y_pred)




