"""
Markov Chain
= 마르코프 성질(Markov Property)을 지닌 이산 확률 과정(Discrete-time Stochastic Process)을 의미한다.

- 마르코프 성질
마르코프 성질이라 함은 n+1회의 상태(state)는 오직 n회에서의 상태, 혹은 그 이전 일정 기간의 상태에만 영향을 받는 것을 의미한다.
예를 들면 동전 던지기는 독립 시행이기 때문에 n번째의 상태가 앞이던지 뒤이던지 간에 n+1번째 상태에 영향을 주지 않는다.

하지만 1차 마르코프 체인은 n번째 상태가 n+1번째 상태를 결정하는데 영향을 미친다.
(시간 t에서의 관측은 단지 단지 최근 r개의 관측에만 의존한다는 가정을 하고 그 가정하에서 성립)

- 마르코프 모델
마르코프 모델은 1차 마르코프 성질의 가정하에 확률적 모델을 만든 것으로써, 가장 먼저 각 상태를 정의하게 된다.
상태(state)는 V={v1,···,vm}로 정의하고, m개의 상태가 존재하게 되는 것이다.

- 상태 전이 확률(State transition Probability)
상태 전이 확률이란 각 상태에서 각 상태로 이동할 확률을 말한다.
상태 전이 확률 a_ij는 상태 v_i에서 상태 v_j로 이동할 확률을 의미한다.
"""
import numpy as np


for _ in range(10):
    # choice(): 랜덤 샘플링, p=확률
    current_state = np.random.choice(range(2), p=[0.5, 0.5])
    print(current_state, end=' ')
print()

# transition_probs: 상태 전이 확률
# s0 -> s0, s1, s2, s3으로 상태가 변할 확률
# s1 -> s0, s1, s2, s3으로 상태가 변할 확률
# s2 -> s0, s1, s2, s3으로 상태가 변할 확률
# s3 -> s0, s1, s2, s3으로 상태가 변할 확률
# ...
# ~> transition_probs shape = (현재 상태의 개수, 변한 상태(미래의 상태)의 개수)
transition_probs = [
    [0.7, 0.2, 0.0, 0.1],     # s0 -> s0, s1, s2, s3
    [0.0, 0.0, 0.9, 0.1],     # s1 -> s0, s1, s2, s3
    [0.0, 1.0, 0.0, 0.0],     # s2 -> s0, s1, s2, s3
    [0.0, 0.0, 0.0, 1.0]      # s3 -> s0, s1, s2, s3
]

max_steps = 50  # 상태를 변화시키는 최대 횟수


def print_sequences():
    current_state = 0   # 현재 상태
    for step in range(max_steps):
        print(current_state, end=' ')
        if current_state == 3:
            # 현재 상태가 s3이라면 다른 상태로 전이 할 수 없으므로 break
            break
        # 랜덤 샘플링 ~> 몬테 카를로 메소드 사용
        # 4가지의 상태(s0, s1, s2, s3), 확률 p=transition_probs[현재상태]
        current_state = np.random.choice(range(4), p=transition_probs[current_state])
    else:
        # for loop가 break를 만나지 않고 전체를 모두 반복했을 때
        print(' ... ')
    print()


if __name__ == '__main__':
    # 위와 같은 상태 전이 확률에서 20번동안 상태는 어떻게 바뀌는가?
    for i in range(20):
        print_sequences()

