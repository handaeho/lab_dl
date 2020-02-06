"""
Policy Gradient(정책 경사법)
= 가장 많은 reward를 받는 방향으로 policy을 정하고, 그에 따라 gradient를 결정하자.

신경망의 파라미터들을 즉각적으로 업데이트 하는 것이 아닌,
여러 에피소드를 진행하고 더 좋은 reward가 발생한 action이 더 높은 확률로 발생할 수 있도록 파라미터를 업데이트 하는 것이다.

각 상황에서 보상이 가장 큰 행동이 무엇인지를 찾는 일 자체도 물론 어렵지만, 그것만으로는 부족하다.
문제의 정답만 외우는 것과 같아서 조금만 다른 상황이 와도 제대로 대응할 수 없기 때문이다.

또한 지금의 어떠한 행동이 결과에 대해 어떤 영향을 미쳤는지 알수 없는 경우가 더 많다.
예를 들어 오목에서 이 한 수가 나의 승리 또는 패배에 영향을 주었는지 등이다.

따라서 상황과 행동과 보상의 관계를 학습해야 한다.

따라서 주어진 상황에서 행동의 보상을 예측하는 모델을 먼저 학습시킨 뒤
실전에서는 가능한 모든 행동에 대한 보상 예측값을 비교하여 최적의 행동을 선택하는 것이다.

‘A라는 상황에서는 B로 행동해’라고 직접적으로 답을 주는 함수가 있다면 좋을 것이다.
이런 방침을 우리는 정책(Policy)이라고 부른다.

‘A이면 무조건 B’라고 결정해놓는 대신 ‘A일 때 70%의 확률로 B0, 나머지 30%는 B1으로 행동해’처럼
정책을 확률적으로 표현하면 보다 일반적인 경우에도 사용할 수 있다. (가위바위보 게임을 생각해보라.)

결국 정책은 상황(S, State)에 대한 행동(A, Action)의 조건부확로 표현할 수 있다.

바로 이것이 'Policy Gradient(정책 경사법)'의 개념이다.

정책은 πθ(A|S)로, 상태 s에서 행동 a를 취했을 때 받는 보상이 R(a, s).일 때,
우리의 목표는 주어진 상황에서의 기대 보상(R, Reward)을 최대로 하는 파라미터 θ를 찾는 것이다.
~~~> J(θ) = E_πθ * [R|S]

목표 함수를 (최소화가 아니라) 최대화해야 하므로 파라미터 업데이트 식은 경사값을 (빼는 것이 아니라) 더한다.
~~~> θn+1 = θn + η * (∂J(θ) / ∂θ)

만약, 취할 수 있는 행동의 범위가 연속적인 실수인 경우는 어떻게 하면 될까?
한 가지 방법은 정책 πθ(a|s)가 내놓는 행동의 확률분포가 정규분포라고 가정하는 것이다.

상태 s가 주어지면, 파라미터 θ에 의해서 정규분포의 평균μ와 표준편차σ를 구한다.
이 분포로부터 값을 하나 샘플링하면, 그게 바로 현재 정책에 따라서 행동을 선택한 것이다.

그에 따라 행동하고 보상받고 θ를 업데이트하는 나머지 과정은 동일하다
"""
import gym
import numpy as np
import tensorflow as tf

from Reinforcement_Learning.ex04_Cartpole_NN import render_policy_net
from tensorflow import keras


def play_one_step(env, obs, model, loss_fn):
    """
    주어진 신경망 모델을 이용해서 게임을 1 step 진행하는 함수

    1 step 진행 후, 바뀐 obs(관측값), reward, done(게임 종료 여부), gradients들을 return
    """
    with tf.GradientTape() as tape:
        # 에측값 left_prob
        left_prob = model(obs[np.newaxis]) # obs를 1d에서 2d로

        action = (tf.random.uniform([1, 1]) > left_prob) # action ~> True or False
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
        # ~> action을 float32 타입으로 바꾼다(True=1, False=0). 그리고 1에서 해당 값을 뺀다.
        # 그럼 우리의 정책에서 결정된 값인 y_target은 0(왼쪽 이동) 또는 1(오른쪽 이동)이 된다.

        # 실제값과 예측값으로 loss 계산
        loss = tf.reduce_mean(loss_fn(y_target, left_prob))

    # 구한 loss로 gradient 계산
    grads = tape.gradient(loss, model.trainable_variables)

    # 게임을 진행하며 나온 결과들(obs(관측값), reward, done(결과), info) ~> 2차원 배열 int 타입의 action으로 게임 진행
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))

    return obs, reward, done, grads


def ply_multiple_episodes(env, n_episodes, max_steps, model, loss_fn):
    """
    여러번의 에피소드를 플레이하는 함수

    모든 에피소드가 종료 된 후, 발생한 모든 reward와 gradients를 return
    """
    all_rewards = [] # 에피소드가 끝날때마다 총 reward를 append할 리스트
    all_grads = [] # 에피소드가 끝날때마다 계산된 gradient를 append할 리스트

    # 지정된 횟수만큼 에피소드 반복
    for episodes in range(n_episodes):
        current_rewards = [] # 각 stpe마다 받은 reward를 append할 리스트
        current_grads = [] # 각 step마다 계산한 gradient를 append할 리스트
        obs = env.reset() # 에피소드마다 새로운 게임을 해야하므로 게임 환경 초기화

        # 정해진 step만큼 한 게임씩 진행하며 각 결과값 계산
        for step in range(max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward) # 한번의 step에서의 reward 저장
            current_grads.append(grads) #  한번의 step에서의 gradient 저장
            if done: # 게임 결과가 True면(게임 종료)
                break
        all_rewards.append(current_rewards) # 모든 step에서 받은 reward 저장
        all_grads.append(current_grads) # 모든 step에서 계산된 gradient 저장

    return all_rewards, all_grads


def discount_reward(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        # ~> -2를 하는 이유는 배열의 가장 마지막 원소는 discount를 할 필요가 없기 때문에 그보다 하나 전의 원소까지만 discount
        discounted[step] += discount_rate * discounted[step + 1]

    return discounted


def discount_normalize_rewards(all_rewards, discount_rate):
    # 모든 reward들에서 하나씩 꺼내 discount_rate에 따라 적용
    all_dc_rewards = [discount_reward(rewards, discount_rate) for rewards in all_rewards]

    # Normalize: z = (x - mean) / std
    flat_rewards = np.concatenate(all_dc_rewards) # normalize를 위해 2차원 all_dc_rewards을 1차원으로
    rewards_mean = flat_rewards.mean() # 1차원으로 변환한 배열의 mean 계산
    rewards_std = flat_rewards.std() # 1차원으로 변환한 배열의 std 계산

    # 모든 원소에 대해 Normalize
    normalize_x = [(x - rewards_mean) / rewards_std for x in all_dc_rewards]

    return normalize_x


if __name__ == '__main__':
    # 테스트
    rewards = [10, 0, -50]

    discounted = discount_reward(rewards, discount_rate=0.8)
    print(discounted)
    # [-22 -40 -50]
    # ~> 마지막 원소는 discount 안되므로 -50,
    # 두번째 원소는 마지막 원소(-50)에 대해 0.8 discount = -40 + 0(원래 값) = -40
    # 첫번째 원소는 두번째 원소(-40)에 대해 0.8 discount = -32 + 10(원래 값) = -22

    all_rewards = [[10, 0, -50],
                   [10, 20]]

    dc_normalized = discount_normalize_rewards(all_rewards, discount_rate=0.8)
    print(dc_normalized)
    # [array([-0.28435071, -0.86597718, -1.18910299]), array([1.26665318, 1.0727777])]

    # 'Policy Gradients'에서 사용한 신경망 모델 생성
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, activation='elu', input_shape=(4, )))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.binary_crossentropy

    # 학습에 필요한 상수들
    n_iterations = 150          # 전체 반복 횟수
    n_episode_per_update = 10   # 신경망 모델을 업데이트하기 전에 실행할 에피소드 횟수
    max_steps = 200             # 한 에피소드에서 실행할 최대 step
    discount_rate = 0.95        # discount_rate ~> 각 step에서의 reward의 discount를 계산

    # 게임 환경 설정
    env = gym.make('CartPole-v1')

    # 게임 반복 실행
    for iteration in range(n_iterations):
        all_rewards, all_grads = ply_multiple_episodes(env=env, n_episodes=n_episode_per_update,
                                                       max_steps=max_steps, model=model, loss_fn=loss_fn)
        # ~> 파라미터를 바로 업데이트 하지않고 한번의 에피소드당, n_episode_per_update 단위만큼 실행하며 일단 '기록'만 한다.

        # 모든 reward 합계와 평균 계산
        total_rewards = sum(map(sum, all_rewards))
        mean_rewards = total_rewards / n_episode_per_update
        print(f'Iteration #{iteration}: mean_rewards={mean_rewards}')

        # reward에 normalize한 discount 적용
        all_final_rewards = discount_normalize_rewards(all_rewards, discount_rate=discount_rate)

        # 모든 gradients의 평균을 저장할 리스트
        all_mean_grads = []

        # 학습하며 gradients 계산
        for idx in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean([final_reward * all_grads[episode_index][step][idx]
                                         for episode_index, final_rewards in enumerate(all_final_rewards)
                                         for step, final_reward in enumerate(final_rewards)],
                                        axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    env.close()

    # 게임 렌더링
    render_policy_net(model, max_steps=1000)


