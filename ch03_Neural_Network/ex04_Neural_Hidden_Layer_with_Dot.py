"""
Neural Network에서의 Dot Product 2

input x의 가중치 W에 대해 'Dot Product'를 사용하면, output y를 쉽계 계산할 수 있다.

어떤 신경망이 input x의 output y에 대하여 2개의 hidden layer를 갖는 3층 구조라고 할 때,
input x(x1, x2) -> 1층 hidden_layer_1 (3개 노드) -> 2층 hidden_layer_2 (2개 노드) -> output y(y1, y2)
한 층의 출력은 다음 층의 입력이 된다. 그리고 각 hidden_layer에서는 'activation function'이 동작한다.
"""
from ch03_Neural_Network.ex01_Activation_Function import sigmoid_function

import numpy as np

# STEP 1) input_layer -> hidden layer 1
# hidden_layer_1 a = x @ W1 + b1
x = np.array([1, 2])
W1 = np.array([[1, 2, 3],
             [4, 5, 6]])
b1 = np.array([1, 2, 3])
a1 = x.dot(W1) + b1
print(a1) # [10 14 18]

# hidden_layer_1 a1에 Sigmoid activation function을 적용한 output z1
z1 = sigmoid_function(a1)
print('hidden 1 =', z1) # hidden 1 = [0.9999546 0.99999917 0.99999998]

# STEP 2) hidden layer 1 -> hidden layer 2 : hidden_layer_2의 input은 hidden_layer_1의 output이다.
# hidden layer_2 a2 = z1 @ W2 + b2
W2 = np.array([[0.1, 0.4],
              [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
a2 = z1.dot(W2) + b2
print(a2) # [0.69999529 1.69998142]

# hidden_layer_2 a2에 Sigmoid activation function을 적용한 output z2
z2 = sigmoid_function(a2)
print('hidden 2 =', z2) # hidden 2 = [0.66818673 0.84553231]

# STEP 3) hidden layer 2 -> output_layer : output_layer의 input은 hidden_layer_2의 output이다.
# output_layer y = z2 @ W3 + b3
W3 = np.array([[0.1, 0.3],
              [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
y = z2.dot(W3) + b3
print('output =', y) # output = [0.33592513 0.73866894]
