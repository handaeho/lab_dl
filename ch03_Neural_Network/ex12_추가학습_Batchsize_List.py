"""
추가 학습
"""
import numpy as np

a = np.arange(10)
print('a =', a) # a = [0 1 2 3 4 5 6 7 8 9]

size = 5
for i in range(0, len(a), size):
    print(a[i:i+size])
    # [0 1 2 3 4]
    # [5 6 7 8 9]

# 파이썬의 리스트
b = [1, 2]
c = [1, 2, 3]
b.append(c)
print(b) # [1, 2, [1, 2, 3]] ~> append()는 1차원 리스트 c가 그대로 append되어 b는 2차원 리스트가 된다.

# numpy의 리스트
x = np.array([1, 2])
y = np.array([3, 4, 5])
x = np.append(x, y)
print(x) # [1 2 3 4 5] ~> np.append()는 리스트를 풀어서 append해서 리스트의 차원이 1차원으로 유지된다.
