"""
Monte Carlo Method(몬테 카를로 메소드)
= 난수를 생성해서 어떤 값을 '확률적'으로 계산하는 방법
"""

""" 반지름이 1인 원 안에 어떤 점 P(x, y)가 들어갈 확률 """
import math
import random

n_iteration = 10000   # 전체 반복 횟수
n_in = 0    # P(x, y)가 반지름이 1인 원 안에 들어간 개수

for _ in range(n_iteration):
    # 난수 x, y 생성
    x = random.random()
    y = random.random()

    # 원점부터 점 P(x, y)의 거리
    d = math.sqrt(x**2 + y**2)

    # 거리 d가 반지름 1보다 작으면 점 P(x, y)가 원 안에 들어간 것이므로 n_in 증가
    if d <= 1.0:
        n_in += 1

# 원주율을 구하기 위해 (n_in / n_iteration)를 하고, 좌표평면은 4사분면이므로 4를 곱한다.
estimate_pi = (n_in / n_iteration) * 4
print(estimate_pi)
# ~> 반복하는 횟수가 증가할수록, '3.141592...'에 근사함을 확인할 수 있다.

