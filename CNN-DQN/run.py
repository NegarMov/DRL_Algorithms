import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import math

from DQN import DQN
from replay_buffer import ReplayBuffer
from wrappers import FireResetEnv, EpisodicLifeEnv

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, ClipReward

import torch

MODEL_PATH = '../_models'


def create_env(params, episodic_life=True, clip_rewards=False):
    gym.register_envs(ale_py)

    env = gym.make(
        params['env_name'], 
        render_mode="human" if params['eval'] else None,
        frameskip=1
    )

    # Wrap the environment to make it suitable for Atari games
    env = AtariPreprocessing(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=params['frame_size'], 
        grayscale_obs=True,
        scale_obs=True
    )
    env = FrameStackObservation(env, stack_size=4)
    if clip_rewards:
        env = ClipReward(env, -1, 1)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    ac_dim = env.action_space.n

    return env, ac_dim


def train(params):
    # Create the environment and get its properties
    env, ac_dim = create_env(params)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(params['max_buffer_size'], (4, params['frame_size'], params['frame_size']))

    # Initialize the networks and copy the weights to target network
    policy_dqn = DQN(ac_dim, params['lr'])
    target_dqn = DQN(ac_dim)

    target_dqn.load_state_dict(policy_dqn.state_dict())
    
    # Array to store aggregated rewards
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])

    update_count = 0
    actions_taken = 0

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False

        ###################
        ### COLLECT A PATH
        ###################

        while not episode_over:
            # Compute the epsilon value for this step
            e_threshold = min(params['e_min'] + (params['e'] - params['e_min']) * \
                math.exp(-1. * (actions_taken - params['warmup_steps']) / params['e_decay_rate']), 1)
            
            # Use epsilon-greedy exploration to choose an action
            if random.rand() < e_threshold or actions_taken < params['warmup_steps']:
                ac = env.action_space.sample()
            else:
                with torch.no_grad():
                    ac = policy_dqn(ob).argmax().item()

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Add the transition to raplay buffer
            replay_buffer.append((ob, ac, new_ob, reward, terminated)) 

            # Collect the reward for this step
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation and add to the number of actions taken
            ob = new_ob
            actions_taken += 1

            # Check if the episode is over
            episode_over = terminated or truncated

        ###################
        ### DQN UPDATE
        ###################

        if len(replay_buffer) > params['batch_size'] and actions_taken > params['warmup_steps']:
            obs, acs, next_obs, rewards, terminateds = replay_buffer.sample(params['batch_size'])

            with torch.no_grad():
                if params['ddqn']:
                    policy_acs = policy_dqn(next_obs).argmax(dim=1)
                    targets = rewards + params['df'] * target_dqn(next_obs).gather(1, policy_acs.unsqueeze(1)).squeeze(1)
                else:
                    targets = rewards + params['df'] * target_dqn(next_obs).max(dim=1)[0]

                # If terminated, target is just reward
                terminateds = terminateds.squeeze()
                targets[terminateds] = rewards[terminateds]  

            # Get Q values for the actions taken
            q_values = policy_dqn(obs).gather(1, acs.unsqueeze(1)).squeeze(1)

            # Calculate loss
            loss = policy_dqn.loss_fn(q_values, targets)

            # Compute the loss and backpropagate
            policy_dqn.optimizer.zero_grad()
            loss.backward()
            policy_dqn.optimizer.step()

            update_count += params['batch_size']

            # Copy policy network weights to target network
            if update_count > params['network_sync_rate']:
                target_dqn.load_state_dict(policy_dqn.state_dict())

        if (e + 1) % params['log_freq'] == 0:
            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            print(f"\nEpisode {e+1}{' (warmup)' if actions_taken < params['warmup_steps'] else ''}: Reward = {reward_per_episodes[e//params['log_freq']]} - Buffer size: {len(replay_buffer)} - epsilon: {e_threshold:.4f}")

    env.close()

    # Plot the results
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes)
    plt.xlabel('Episode Number')
    plt.ylabel(f"Average Reward per {params['log_freq']} Episodes")
    plt.show()

    # Save the learned policy network
    print('\nSaving policy network...')
    out_name = f"{params['env_name'].split('/')[-1]}_CNN-DQN"
    policy_dqn.save(f"{MODEL_PATH}/{out_name}.pt")


def evaluate(params):
    # Create the environment and get its properties
    env, ac_dim = create_env(params)

    # Initialize and load the policy network
    print('\nLoading policy network...\n')
    policy = DQN(ac_dim)
    out_name = f"{params['env_name'].split('/')[-1]}_CNN-DQN"
    policy.load(f"{MODEL_PATH}/{out_name}.pt")

    # Run episodes
    for e in range(params['episodes']):
        # Reset the environment for a new episode
        ob, _ = env.reset()
        print('reset')

        episode_over = False
        episode_reward = 0

        while not episode_over:
            # Sample an action from the policy
            with torch.no_grad():
                ac = policy(ob).argmax().item()

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Collect the reward for this step
            episode_reward += reward

            # Update the observation and go for the next step
            ob = new_ob

            # Check if the episode is over
            episode_over = terminated or truncated

        print(f"Episode {e+1}: Reward = {episode_reward}")
                
    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='ALE/Pong-v5', help='Environment name')
    parser.add_argument('--frame_size', type=int, default=84, help='The size to which the frames would be resized to')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--warmup_steps', type=int, default=50000, help='Number of steps before training start')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--network_sync_rate', type=int, default=10000, help='Frequency at which policy and target networks are synced')
    parser.add_argument('--max_buffer_size', type=int, default=int(1e6), help='Maximum capacity of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of experiences sampled from the replay buffer for each training iteration')
    parser.add_argument('--df', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--e', type=float, default=1.0, help='The starting value for epsilon')
    parser.add_argument('--e_min', type=float, default=0.0, help='The minimum value for epsilon')
    parser.add_argument('--e_decay_rate', type=float, default=0.0, help='Epsilon decay rate')
    parser.add_argument('--ddqn', action='store_true', help='Use double DQN')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Run the main function
    if params['eval']:
        evaluate(params)
    else:
        train(params)