"""
.dim -> 차원
.shape -> (행, 열). 1차원의 경우(원소의 개수, )
.size -> 전체 원소의 개수
.len -> 길이
"""
import numpy as np

a = np.array([1, 2, 3]) # 1차원 리스트 a
print('dim a =', a.ndim) # dim a = 1
print('shape a =', a.shape) # shape a = (3,)
print('size a=', a.size) # size a = 3 ~> 전체 원소의 개수
print('len  a =', len(a)) # len a = 3

A = np.array([[1, 2, 3],
              [4, 5, 6]]) # 2차원 리스트 A
print('dim A =', A.ndim) # dim A = 2
print('shape A =', A.shape) # shape A = (2, 3)
print('size A =', A.size) # size A = 6
print('len A =', len(A)) # len A = 2