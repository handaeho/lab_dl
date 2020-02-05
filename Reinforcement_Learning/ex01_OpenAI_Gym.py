"""
Must need to setup 'pip install gym'

OpenAI인 'Gym'의 'CartPole-v1' 게임으로 'Gym 라이브러리'와 'Reinforcement Learning' 이해하기

카트 위의 막대를 똑바로 세우는 게임으로 카트가 이동하며 막대가 흔들린다.
이 게임에는 '관성' 이 적용되므로 카트를 움직여가며 막대를 세울수 있는 카트의 속도와 위치를 찾는 게임
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 1. Gym 라이브러리 버전
    print(gym.__version__)

    # 2. gym 패키지 환경 리스트
    print(gym.envs.registry.all())
    # dict_values([EnvSpec(Copy-v0), EnvSpec(RepeatCopy-v0), EnvSpec(ReversedAddition-v0), ...

    # 3. CartPole-v1 게임 환경 생성
    env = gym.make('CartPole-v1')

    # 4. Environment 초기화
    obs = env.reset() # observations(관찰값, 관측값)
    print(obs)
    # [0.03940892 -0.04371548 -0.04949015 -0.04333724] ~> [카트 위치(x축), 카트의 속도, pole의 각도, pole의 각속도]

    # 5. Environment 시각화
    env.render()

    # 6. Environment Rendering Image 저장
    img = env.render(mode='rgb_array')
    print(img)
    print(img.shape)
    # (400, 600, 3) ~> [height, width, color(RGB)]

    # 7. matplotlib으로 Image 시각화
    plt.imshow(img)
    plt.show()

    # 8. action: 게임 실행, 게임 상태 변경
    # 8-1) 가능한 action의 개수
    print(env.action_space)
    # Discrete(2) ~> 불연속적인 action. 0 또는 1의 action을 가짐.
    # action 0 ~> 왼쪽 방향(-)으로 가속도
    # action 1 ~> 오른쪽 방향(+)으로 가속도

    # 8-2) action 종류 설정 및 게임 상태 변경
    action = 1
    obs, reward, done, info = env.step(action) # env.action(action 상태) ~> 'obs, reward, done, info' 4개를 리턴
    print(obs) # [-0.02418954  0.21136392 -0.04341626 -0.28366341] ~> 카트와 pole의 상태가 변화함.
    print(reward) # 이전 행동으로 달성 한 보상 금액. 규모는 환경에 따라 다르지만 목표는 항상 총 보상을 높이는 것.
    print(done) # 결과
    print(info) # 디버깅에 유용한 진단 정보(환경의 마지막 상태 변경에 대한 원시 확률 등)

    # 사용한 게임 Environment 종료
    env.close()







