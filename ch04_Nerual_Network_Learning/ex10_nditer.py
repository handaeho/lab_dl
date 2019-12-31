"""
numpy.nditer에 대해 알아보자.

numpy.nditer 객체: 반복문(for, while)을 사용하기 쉽게 도와주는 역할을 하는 객체
"""
import numpy as np

np.random.seed(1231)

# 2x3 100미만의 난수 행렬 a
a = np.random.randint(100, size=(2, 3))
print(a)

# a의 원소들을 '40 21 5 52 84 39'의 형태로 a를 출력하려면
for row in a:
    for x in row:
        print(x, end=' ')
print()

# 같은 결과를 while문으로
i = 0
while i < a.shape[0]: # i < row 개수
    j = 0
    while j < a.shape[1]: # j < column 개수
        print(a[i, j], end=' ')
        j += 1
    i += 1
print()

# 이번에는 같은 과정을 np.nditer를 사용해 진행하자
# 먼저 nditer 객체 생성
with np.nditer(a) as iterator: # with ~ as : 기능이 끝나면 자동으로 종료해 줄 때 사용
    for val in iterator:
        print(val, end=' ')
print()

# while문에 nditer 객체를 적용하면
with np.nditer(a, flags=['multi_index']) as iterator:
    while not iterator.finished: # iterator.finished -> 반복이 끝나면 True, 끝나지 않으면 False
        i = iterator.multi_index # multi_index -> (행의 인덱스, 열의 인덱스). 사용하려면 먼저 'flags=['multi_index']' 필요
        print(f'{i}, {a[i]}', end=' ')
        iterator.iternext() # 다음 순서로
print()
# ~> (0, 0), 40 (0, 1), 21 (0, 2), 5 (1, 0), 52 (1, 1), 84 (1, 2), 39

# 'c_index'를 사용한 다른 버전
with np.nditer(a, flags=['c_index']) as iterator: # c_index는 리스트를 flatten해서 0번부터 인덱스를 가져온다
    while not iterator.finished:
        i = iterator.index
        print(f'{i}: {iterator[0]}', end=' ') # 그래서 0번부터 시작해서 증가 해야한다.
        iterator.iternext()
print()
# ~> 0: 40 1: 21 2: 5 3: 52 4: 84 5: 39

# 그럼 np.nditer를 사용해 리스트 a의 값을 변경 할 수 있을까?
a = np.arange(6).reshape((2, 3))
print(a)
# [[0 1 2]
#  [3 4 5]]

with np.nditer(a, flags=['multi_index']) as it:
    while not it.finished:
        a[it.multi_index] *= 2
        it.iternext()
print(a)
# [[ 0  2  4]
#  [ 6  8 10]]

# 그러나 'c_index'는 리스트의 값을 변경 할 수 없다.
a = np.arange(6).reshape((2, 3))
print(a)
# [[0 1 2]
#  [3 4 5]]

with np.nditer(a, flags=['c_index'], op_flags=['readwrite']) as it:
    while not it.finished:
        it[0] *= 2
        it.iternext()
print(a)
# ValueError: output array is read-only 발생 ~> index만 사용하는 'c_index'는 'read only'이다.
# 따라서, op_flags=['readwrite']를 반드시 추가해 wirte도 가능하게 만들어 주어야 한다.







