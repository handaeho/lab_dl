"""
Neural Network에서의 Dot Product 1

input x의 가중치 W에 대해 'Dot Product'를 사용하면, output y를 쉽계 계산할 수 있다.
"""
import numpy as np


x1 = np.array([1, 2])
W1 = np.array([[1, 4],
              [2, 5],
              [3, 6]])
b1 = 1
y1 = W1.dot(x1) + b1 # np.dot(x, W) + b
print(y1)

x2 = np.array([1, 2])
W2 = np.array([[1, 2, 3],
             [4, 5, 6]])
b2 = 1
y2 = x2.dot(W2) + b2 # # np.dot(W, x) + b
print(y2)


