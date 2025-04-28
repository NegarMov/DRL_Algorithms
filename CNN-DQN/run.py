import numpy as np
from numpy import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from DuelingDQN import DuelingDQN
from replay_buffer import ReplayBuffer

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation, ClipReward
from wrappers import FireResetEnv, EpisodicLifeEnv

import torch

MODEL_PATH = '../_models'


def create_env(params, episodic_life=False, clip_rewards=False, fire_reset=False):
    gym.register_envs(ale_py)

    env = gym.make(
        params['env_name'], 
        max_episode_steps=100000,
        render_mode="human" if params['eval'] else None
    )

    # Wrap the environment to make it suitable for Atari games
    env = AtariPreprocessing(
        env,
        noop_max=0,
        scale_obs=True,
        grayscale_obs=True,
        screen_size=80
    )

    env = FrameStackObservation(env, 4)
    
    env = TransformObservation(
        env, 
        lambda obs: obs[:, 14:78, :], 
        env.observation_space
    )

    # Game-specific wrappers
    if clip_rewards:
        env = ClipReward(env, -1, 1)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    if fire_reset:
        env = FireResetEnv(env)
    
    ac_dim = env.action_space.n

    return env, ac_dim


def train(params):
    # Create the environment and get its properties
    env, ac_dim = create_env(params)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(params['max_buffer_size'])

    # Initialize the networks and copy the weights to target network
    policy_dqn = DuelingDQN(ac_dim, params['lr'])
    target_dqn = DuelingDQN(ac_dim, params['lr'])
    target_dqn.load_state_dict(policy_dqn.state_dict())
    target_dqn.eval()
    
    # Array to store aggregated rewards
    reward_per_episodes = deque(maxlen=100)

    actions_taken = 0

    e_threshold = params['e']

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False
        episode_reward = 0
        episode_loss = 0
        
        ###################
        ### COLLECT A PATH
        ###################

        while not episode_over:
           # Use epsilon-greedy exploration to choose an action
            if random.rand() <= e_threshold:
                ac = env.action_space.sample()
            else:
                with torch.no_grad():
                    ac = policy_dqn(ob).argmax().item()

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Add the transition to raplay buffer
            replay_buffer.append((ob, ac, new_ob, reward, terminated)) 

            # Collect the reward for this step
            episode_reward += reward

            # Update the observation and add to the number of actions taken
            ob = new_ob
            actions_taken += 1

            # Check if the episode is over
            episode_over = terminated or truncated

            # Compute the epsilon value for the next step
            if actions_taken % 1000 == 0:
                e_threshold = max(e_threshold * params['e_decay_rate'], params['e_min'])

            ###################
            ### DQN UPDATE
            ###################

            if len(replay_buffer) >= params['warmup_steps']:
                obs, acs, next_obs, rewards, terminateds = replay_buffer.sample(params['batch_size'])

                with torch.no_grad():
                    policy_acs = policy_dqn(next_obs).max(1)[1].unsqueeze(1)
                    targets = rewards + params['df'] * target_dqn(next_obs).gather(1, policy_acs).squeeze(1)

                    # If terminated, target is just reward
                    targets[terminateds] = rewards[terminateds]

                policy_values = policy_dqn(obs).gather(1, acs.unsqueeze(1)).squeeze(1)

                # Compute the loss and backpropagate
                loss = (policy_values - targets).pow(2).mean()

                policy_dqn.optimizer.zero_grad()
                loss.backward()
                policy_dqn.optimizer.step()

                episode_loss += loss.item()

        # Copy policy network weights to target network
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Log episode metrics
        reward_per_episodes.append(episode_reward)
        print(f"\nEpisode {e+1}{' (warmup)' if actions_taken < params['warmup_steps'] else ''} -- Reward: {episode_reward} - Loss: {episode_loss:.3f} - Last 100 avg reward: {np.mean(reward_per_episodes):.2f} - Epsilon: {e_threshold:.4f} - Buffer size: {len(replay_buffer)}")

        # Save the model as a checkpoint
        if (e + 1) % params['save_freq'] == 0:
            out_name = f"{params['env_name'].split('/')[-1]}_CNN-DQN_{e+1}"
            policy_dqn.save(f"{MODEL_PATH}/{out_name}.pt")

    env.close()

    # Plot the results
    plt.plot(reward_per_episodes)
    plt.xlabel('Episode Number')
    plt.ylabel(f"Episode Return")
    plt.show()

    # Save the final policy network
    print('\nSaving policy network...')
    out_name = f"{params['env_name'].split('/')[-1]}_CNN-DQN"
    policy_dqn.save(f"{MODEL_PATH}/{out_name}.pt")


def evaluate(params):
    # Create the environment and get its properties
    env, ac_dim = create_env(params)

    # Initialize and load the policy network
    print('\nLoading policy network...\n')
    policy = DuelingDQN(ac_dim)
    out_name = f"{params['env_name'].split('/')[-1]}_CNN-DQN_{params['checkpoint']}"
    policy.load(f"{MODEL_PATH}/{out_name}.pt")

    # Run episodes
    for e in range(params['episodes']):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False
        episode_reward = 0

        while not episode_over:
            # Choose an action
            if random.rand() <= params['e']:
                ac = env.action_space.sample()
            else:
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
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--warmup_steps', type=int, default=50000, help='Number of steps before training start')
    parser.add_argument('--save_freq', type=int, default=100, help='Frequency at which the model is saved (in episodes)')
    parser.add_argument('--max_buffer_size', type=int, default=int(1e6), help='Maximum capacity of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of experiences sampled from the replay buffer for each training iteration')
    parser.add_argument('--df', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--e', type=float, default=1.0, help='The starting value for epsilon')
    parser.add_argument('--e_min', type=float, default=0.0, help='The minimum value for epsilon')
    parser.add_argument('--e_decay_rate', type=float, default=0.0, help='Epsilon decay rate')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--checkpoint', type=int, default=100, help='The checkpoint at which evaluation is done')
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    # Run the main function
    if params['eval']:
        evaluate(params)
    else:
        train(params)