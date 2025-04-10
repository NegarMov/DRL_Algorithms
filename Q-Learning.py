import gymnasium as gym
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pickle


MODEL_PATH = './_models/Q-Learning_FrozenLake.pkl'


def run(episodes, eval=False):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if eval else None)

    lr = 1                  # learning rate
    df = 0.9                # discount factor
    e = 1                   # epsilon
    e_decay_rate = 1e-4     # epsilon decay rate
    aggregation_interval = 1 if eval else 100

    if eval:
        f = open(MODEL_PATH, 'rb')
        q_table = pickle.load(f)
        f.close()
    else:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    reward_per_episodes = np.zeros(episodes//aggregation_interval)

    for i in range(episodes):
        observation, _ = env.reset()
        episode_over = False

        while not episode_over:
            if not eval and random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation, :])

            new_observation, reward, terminated, truncated, _ = env.step(action)

            if not eval:
                q_table[observation, action] = \
                    q_table[observation, action] + \
                    lr * (reward + df * np.max(q_table[new_observation, :]) - q_table[observation, action])

            observation = new_observation
            reward_per_episodes[i//aggregation_interval] += reward
            episode_over = terminated or truncated

        e = max(e - e_decay_rate, 0)

    env.close()

    if not eval:
        plt.plot(np.arange(0, episodes, 100), reward_per_episodes / 100)
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward per 100 Episodes')
        plt.show()

        f = open(MODEL_PATH,"wb")
        pickle.dump(q_table, f)
        f.close()


if __name__ == '__main__':
    run(10000)
    run(1, eval=True)