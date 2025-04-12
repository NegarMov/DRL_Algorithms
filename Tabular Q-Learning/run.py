import gymnasium as gym
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm


MODEL_PATH = '../_models'


def create_env(params):
    env = gym.make(
        params['env_name'], 
        map_name="8x8", 
        is_slippery=params['is_slippery'],
        render_mode="human" if params['eval'] else None
    )

    ob_dim = env.observation_space.n
    ac_dim = env.action_space.n

    return env, ob_dim, ac_dim


def train(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize the Q table
    q_table = np.zeros((ob_dim, ac_dim))
    
    # Array to store aggregated rewards
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])

    # Store the user-specified learning rate
    initial_lr = params['lr']

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        observation, _ = env.reset()

        episode_over = False

        while not episode_over:
            # Use epsilon-greedy exploration to choose an action
            if random.rand() < params['e']:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation, :])

            # Run the action on the environment
            new_observation, reward, terminated, truncated, _ = env.step(action)

            # Update the Q values
            y = reward + params['df'] * np.max(q_table[new_observation, :])
            q_table[observation, action] -= params['lr'] * (q_table[observation, action] - y)

            # Collect the reward for this step
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation
            observation = new_observation

            # Check if the episode is over
            episode_over = terminated or truncated

        # Decrease epsilon (less exploration)
        params['e'] = max(params['e'] - params['e_decay_rate'], 0)

        # Decrease the learning rate as the learning proceeds
        if (params['e'] == 0):
            params['lr'] = initial_lr * 0.25
        elif (params['e'] < 0.5):
            params['lr'] = initial_lr * 0.5

        if (e + 1) % params['log_freq'] == 0:
            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            print(f"\nEpisode {e+1}: Reward = {reward_per_episodes[e//params['log_freq']]}")

    env.close()

    # Plot the results
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes)
    plt.xlabel('Episode Number')
    plt.ylabel(f"Average Reward per {params['log_freq']} Episodes")
    plt.show()

    # Save the learned Q values
    print('\nSaving Q values...')
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_Q-Learning"
    f = open(f"{MODEL_PATH}/{out_name}.pkl","wb")
    pickle.dump(q_table, f)
    f.close()


def evaluate(params):
    # Create the environment and get its properties
    env, _, _ = create_env(params)

    # Load the Q table
    print('\nLoading Q values...\n')
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_Q-Learning"
    f = open(f"{MODEL_PATH}/{out_name}.pkl", 'rb')
    q_table = pickle.load(f)
    f.close()

    # Run episodes
    for e in range(params['episodes']):
        # Reset the environment for a new episode
        observation, _ = env.reset()
        
        episode_over = False
        episode_reward = 0

        while not episode_over:
            # Choose the best action according to the Q table
            action = np.argmax(q_table[observation, :])

            # Run the action on the environment
            new_observation, reward, terminated, truncated, _ = env.step(action)

            # Collect the reward for this step
            episode_reward += reward
            
            # Update the observation
            observation = new_observation
            
            # Check if the episode is over
            episode_over = terminated or truncated

        print(f"Episode {e+1}: Reward = {episode_reward}")

    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FrozenLake-v1', help='Environment name')
    parser.add_argument('--is_slippery', action='store_true', help='Indicates if the transition probabilities are non-deterministic')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--log_freq', type=int, default=1000, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--df', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--e', type=float, default=1.0, help='Epsilon')
    parser.add_argument('--e_decay_rate', type=float, default=0.0, help='Epsilon decay rate')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    # Run the main function
    if params['eval']:
        evaluate(params)
    else:
        train(params)