"""
OpenAI인 'Gym'의 'CartPole-v1' 게임에 'Policy(정책)'을 적용해보자.
"""
import gym
import numpy as np


def random_policy():
    """
    Policy를 결정하는 함수

    :return: 0 또는 1(카트의 방향. 0 = 왼쪽(-) / 1 = 오른쪽(+))
    """
    return np.random.randint(0, 2)


def basic_policy(obs):
    """
    Environment 관측값(obs)에서 '막대의 각도'에 따라 Policy를 결정하는 함수
    theta > 0 ~> 카트를 오른쪽(+)으로 이동. action = 1
    theta < 0 ~> 카트를 왼쪽(-)으로 이동. action = 0

    :param obs: [x, v, theta, w] = [카트의 위치, 카트의 속도, 막대의 각도, 막대의 각속도]
    :return:
    """
    theta = obs[2]
    if theta > 0:
        aciton = 1
    elif theta < 0:
        aciton = 0

    return aciton


if __name__ == '__main__':
    # 게임 Environment 생성
    env = gym.make('CartPole-v1')

    # Environment 초기화
    obs = env.reset()

    # Rendering(게임 환경 화면 출력)
    env.render()

    # episode: 게임 실행 횟수
    max_episodes = 100
    # 1 에피소드 = Pole이 넘어지기 전까지(done is False)

    # 1 에피소드에서 최대 반복 횟수 설정
    max_steps = 1000

    # =============== Random Policy Ver. ===============
    # 에피소드 횟수만큼 반복
    total_reward = [] # 한 에피소드가 끝날때마다 얻은 보상을 저장할 리스트
    for episode in range(max_episodes):
        print(f'----- RANDOM Episode #{episode+1} -----')
        obs = env.reset() # 한 에피소드가 끝날때마다 Environment 초기화
        episode_reward = 0 # 1 에피소드에서 얻은 reward

        for step in range(max_steps): # 각 에피소드마다 최대 횟수 반복
            env.render() # 게임 화면 렌더링
            aciton = random_policy() # Policy에 따라 액션 선택
            obs, reward, done, info = env.step(aciton) # 게임 상태 변경
            episode_reward += reward # 해당 에피소드의 보상을 계속 더함
            if done is True:
                print(f'*** RANDOM FINISHED! ***')
                break
        total_reward.append(episode_reward) # 한 에피소드가 끝날때마다 얻은 보상을 저장

    print(f'Random Total Reward = {total_reward}')
    # reward의 평균, 표준편차, 최대값, 최소값
    print(f'Random Reward Max:{np.max(total_reward)}, Random Reward Min:{np.min(total_reward)}, '
          f'Random Reward Avg.:{np.mean(total_reward)}, Random Reward Std.:{np.std(total_reward)}')

    # =============== Basic Policy Ver. ===============
    # 에피소드 횟수만큼 반복
    total_reward = []  # 한 에피소드가 끝날때마다 얻은 보상을 저장할 리스트
    for episode in range(max_episodes) :
        print(f'----- BASIC Episode #{episode + 1} -----')
        obs = env.reset()  # 한 에피소드가 끝날때마다 Environment 초기화
        episode_reward = 0  # 1 에피소드에서 얻은 reward

        for step in range(max_steps) :  # 각 에피소드마다 최대 횟수 반복
            env.render()  # 게임 화면 렌더링
            aciton = basic_policy(obs)  # Policy에 따라 액션 선택
            obs, reward, done, info = env.step(aciton)  # 게임 상태 변경
            episode_reward += reward  # 해당 에피소드의 보상을 계속 더함
            if done is True :
                print(f'*** BASIC FINISHED! ***')
                break
        total_reward.append(episode_reward)  # 한 에피소드가 끝날때마다 얻은 보상을 저장

    print(f'Basic Total Reward = {total_reward}')
    # reward의 평균, 표준편차, 최대값, 최소값
    print(f'Basic Reward Max:{np.max(total_reward)}, BasicReward Min:{np.min(total_reward)}, '
          f'Basic Reward Avg.:{np.mean(total_reward)}, Basic Reward Std.:{np.std(total_reward)}')

    # 게임 종료
    env.close()