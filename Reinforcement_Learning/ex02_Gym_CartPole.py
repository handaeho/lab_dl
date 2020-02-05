"""
OpenAI인 'Gym'의 'CartPole-v1' 게임으로 'Gym 라이브러리'와 'Reinforcement Learning' 이해하기 2
"""
import numpy as np
import gym


if __name__ == '__main__':
    # 게임 Environment 생성
    env = gym.make('CartPole-v1')

    # Environment 초기화
    obs = env.reset()

    # Rendering(게임 환경 화면 출력)
    env.render()

    # 최대 반복 횟수 설정
    max_steps = 1000
    for i in range(max_steps):
        action = np.random.randint(0, 2)  # 게임 action 설정(0 = 카트가 왼쪽(-) 이동/ 1 = 카트가 오른쪽(+) 이동)
        obs, reward, done, info = env.step(action)  # action에 따른 게임 상태 변경
        env.render()  # 게임 환경 화면 출력
        print(f'\n action: {action}')
        print(obs)
        print(f'reward: {reward}, done: {done}, info: {info}')
        if done is True:
            break

    # 게임 종료
    env.close()
