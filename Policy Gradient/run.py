import gymnasium as gym
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from policy import Policy

import torch


# Path to save and load model parameters
MODEL_PATH = '../_models'


def create_env(params):
    env = gym.make(params['env_name'], render_mode="human" if params['eval'] else None)

    if isinstance(env.observation_space, gym.spaces.Box):
        ob_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        ob_dim = env.observation_space.n

    if isinstance(env.action_space, gym.spaces.Box):
        ac_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        ac_dim = env.action_space.n
    
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    return env, ob_dim, ac_dim, discrete


def train(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim, discrete = create_env(params)

    # Initialize the policy
    policy = Policy(
        ob_dim, 
        ac_dim, 
        discrete, 
        hidden_layers=params['hidden_layers'], 
        hidden_size=params['hidden_size'], 
        lr=params['lr']
    )
        
    # Arrays to store aggregated rewards and losses
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])
    loss_per_episodes = np.zeros(params['episodes']//params['log_freq'])

    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False
        episode_probs = []
        episode_rewards = []
        steps = 0

        ###################
        ### RUN AN EPISODE
        ###################
        
        while not episode_over:
            # Sample an action from the policy
            ac_dist = policy(ob)
            ac = ac_dist.sample()

            # Store action log probabilities for policy update
            episode_probs.append(ac_dist.log_prob(ac))
            
            # Convert the action to a numpy array
            ac = ac.data.cpu().numpy().astype(env.action_space.dtype)

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Collect the reward for this step
            episode_rewards.append(reward)
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation and go for the next step
            ob = new_ob
            steps += 1

            # Check if the episode is over
            episode_over = terminated or truncated or steps > params['max_episode_len']

        ###################
        ### POLICY UPDATE
        ###################

        # Discount rewards to compute the discounted sum
        discounted_sum = 0
        for i in reversed(range(steps)):
            episode_rewards[i] += discounted_sum
            discounted_sum = params['df'] * episode_rewards[i]

        # Normalize the discounted rewards to reduce variance
        reward_mean = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards) + 1e-7
        episode_rewards = [(r - reward_mean) / reward_std for r in episode_rewards]

        # Compute the policy loss and backpropagate
        policy.optimizer.zero_grad()

        episode_rewards = torch.tensor(episode_rewards, dtype=torch.float32, device=policy.device)
        episode_probs = torch.stack(episode_probs)
        if episode_probs.dim() > 1:
            episode_probs = episode_probs.sum(dim=-1)

        loss = -1 * torch.sum(episode_probs * episode_rewards)

        loss.backward()
        policy.optimizer.step()

        # Collect the loss for this episode
        loss_per_episodes[e//params['log_freq']] += loss.item()

        # Print progress at the end of each aggregation interval
        if (e + 1) % params['log_freq'] == 0:
            loss_per_episodes[e//params['log_freq']] /= params['log_freq']
            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            print(f"\nEpisode {e+1}: Loss = {loss_per_episodes[e//params['log_freq']]}, Reward = {reward_per_episodes[e//params['log_freq']]}")
                
    env.close()

    # Plot the results
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes)
    axs[0].set_xlabel('Episode Number')
    axs[0].set_ylabel(f"Average Reward per {params['log_freq']} Episodes")

    axs[1].plot(np.arange(0, params['episodes'], params['log_freq']), loss_per_episodes)
    axs[1].set_xlabel('Episode Number')
    axs[1].set_ylabel(f"Average loss per {params['log_freq']} Episodes")

    plt.tight_layout()
    plt.show()

    # Save the traied policy
    print('\nSaving policy parameters...')
    policy.save(f"{MODEL_PATH}/{params['env_name']}_policy_gradient.pt")


def evaluate(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim, discrete = create_env(params)

    # Initialize and load the policy
    policy = Policy(
        ob_dim, 
        ac_dim, 
        discrete, 
        hidden_layers=params['hidden_layers'], 
        hidden_size=params['hidden_size'], 
        lr=params['lr']
    )
    print('\nLoading policy parameters...\n')
    policy.load(f"{MODEL_PATH}/{params['env_name']}_policy_gradient.pt")

    # Run episodes
    for e in range(params['episodes']):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False
        episode_reward = 0
        steps = 0
        
        while not episode_over:
            # Sample an action from the policy
            with torch.no_grad():
                ac_dist = policy(ob)
                ac = ac_dist.sample()
                ac = ac.data.cpu().numpy().astype(env.action_space.dtype)

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Collect the reward for this step
            episode_reward += reward

            # Update the observation and go for the next step
            ob = new_ob
            steps += 1

            # Check if the episode is over
            episode_over = terminated or truncated or steps > params['max_episode_len']

        print(f"Episode {e+1}: Reward = {episode_reward}")
                
    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run CartPole-v1 with Policy Gradient')
    parser.add_argument('--env_name', type=str, help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_episode_len', type=int, default=1000, help='Maximum episode length')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--df', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of each hidden layer')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    # Run the main function
    if params['eval']:
        evaluate(params)
    else:
        train(params)
