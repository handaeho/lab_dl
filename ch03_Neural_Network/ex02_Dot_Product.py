"""
행렬의 내적(Dot Product)
- np.dot() : 입력이 1차원 배열이면 '벡터'를, 2차원 배열이면 '행렬 곱'을 계산
             단, np.dot(A, B)를 계산 시, 반드시 '행렬 A의 열 수 = 행렬 B의 행 수'이어야 한다.
             ~~~> A(n, m) @ B(m, l) = X(n, l)
"""
import numpy as np


x = np.array([1, 2]) # 1x2 1차원 행렬 x
y = np.array([[3, 4], [5, 6]]) # 2x2 2차원 행렬 y
result = np.dot(x, y) # 1x2 행렬 result
print(result)

A = np.arange(1, 7)
print(A) # 1x6 1차원 행렬 A
A = A.reshape((2, 3))
print(A) # 2x3 2차원 행렬 A

B = np.arange(1, 7).reshape((3, 2))
print(B) # 3x2 2차원 행렬 B

print(A.dot(B)) # np.dot(B, A) ~> B(3x2) @ A(2x3) = X(2x2)
print(B.dot(A)) # np.dot(A, B) ~> A(2x3) @ B(3x2) = X(3x3)
# ~> 행렬의 내적 연산은 '교환법칙(AB = BA)'이 성립하지 않는다.

# ndarray.shape -> 1차원 (x, ) / 2차원 (x, y) / 3차원 (x, y, z) / ...
x1 = np.array([1, 2, 3])
print(x1)
print(x1.shape) # (3,) ~> 원소의 개수(3개)를 의미


x1 = x1.reshape((3, 1)) # .reshape() : 형 변환
print(x1)
print(x1.shape) # (3, 1) ~> 3행 1열 1차원 행렬

x2 = np.array([[1, 2], [2, 3], [3, 4]])
print(x2)
print(x2.shape) # (3, 2) ~> 3행 2열 2차원 행렬

x2 = x2.reshape((2, 3)) # .reshape() : 형 변환
print(x2)
print(x2.shape) # (2, 3) ~> 2행 3열 2차원 행렬

