import gymnasium as gym
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from DQN import DQN
from DuelingDQN import DuelingDQN
from replay_buffer import ReplayBuffer

import torch


MODEL_PATH = '../_models'


def create_env(params):
    if 'FrozenLake' in params['env_name']:
        env = gym.make(
            params['env_name'],
            map_name="4x4", 
            is_slippery=params['is_slippery'],
            render_mode="human" if params['eval'] else None
        )
    else:
        env = gym.make(
            params['env_name'], 
            render_mode="human" if params['eval'] else None
        )

    if isinstance(env.observation_space, gym.spaces.Box):
        ob_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        ob_dim = env.observation_space.n

    ac_dim = env.action_space.n

    return env, ob_dim, ac_dim


def encode_ob(ob, ob_dim, env):
    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_tensor = torch.zeros(ob_dim)
        input_tensor[ob] = 1
        return input_tensor

    return ob


def train(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(params['max_buffer_size'])

    # Initialize the networks and copy the weights to target network
    if params['dueling']:
        policy_dqn = DuelingDQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['lr'])
        target_dqn = DuelingDQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    else:
        policy_dqn = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['lr'])
        target_dqn = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])

    target_dqn.load_state_dict(policy_dqn.state_dict())
    
    # Array to store aggregated rewards
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])
    val_rewards = np.zeros(params['episodes']//params['log_freq'])

    update_count = 0

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False

        # Compute the epsilon value for this episode
        e_threshold = max(params['e_min'], params['e'] * (params['e_decay_rate'] ** e))

        ###################
        ### COLLECT A PATH
        ###################

        while not episode_over:
            # Use epsilon-greedy exploration to choose an action
            if random.rand() < e_threshold:
                ac = env.action_space.sample()
            else:
                with torch.no_grad():
                    ac = policy_dqn(encode_ob(ob, ob_dim, env)).argmax().item()

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Add the transition to raplay buffer
            replay_buffer.append((ob, ac, new_ob, reward, terminated)) 

            # Collect the reward for this step
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation
            ob = new_ob

            # Check if the episode is over
            episode_over = terminated or truncated

        ###################
        ### DQN UPDATE
        ###################

        if len(replay_buffer) > params['batch_size'] and np.sum(reward_per_episodes) > 0:
            obs, acs, next_obs, rewards, terminateds = replay_buffer.sample(params['batch_size'])

            with torch.no_grad():
                if params['ddqn'] or params['dueling']:
                    policy_acs = policy_dqn(next_obs).argmax(dim=1)
                    targets = rewards + params['df'] * target_dqn(next_obs).gather(1, policy_acs.unsqueeze(1)).squeeze(1)
                else:
                    targets = rewards + params['df'] * target_dqn(next_obs).max(dim=1)[0]

                # If terminated, target is just reward
                targets[terminateds] = rewards[terminateds]  

            target_q_values = target_dqn(obs)
            for i in range(params['batch_size']):
                target_q_values[i, acs[i]] = targets[i]

            # Compute the loss and backpropagate
            policy_dqn.optimizer.zero_grad()

            # Calculate loss
            loss = policy_dqn.loss_fn(policy_dqn(obs), target_q_values)

            loss.backward()
            policy_dqn.optimizer.step()

            update_count += 1

            # Copy policy network weights to target network
            if update_count > params['network_sync_rate']:
                target_dqn.load_state_dict(policy_dqn.state_dict())

        ###################
        ### LOG METRICS
        ###################

        if (e + 1) % params['log_freq'] == 0:
            # Run validation 
            for _ in range(params['val_episodes']):
                val_rewards[e//params['log_freq']] += run_eval_episode(env, ob_dim, policy_dqn)

            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            val_rewards[e//params['log_freq']] /= params['val_episodes']
            print(f"\nEpisode {e+1}: Training reward = {reward_per_episodes[e//params['log_freq']]}, Validation reward: {val_rewards[e//params['log_freq']]}")

    env.close()

    # Plot the results
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes, label="Training Reward")
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), val_rewards, label="Validation Reward")
    plt.xlabel('Episode Number')
    plt.ylabel(f"Average Reward per {params['log_freq']} Episodes")
    plt.legend()
    plt.show()

    # Save the learned policy network
    print('\nSaving policy network...')
    model_type = 'DDQN' if params['ddqn'] else 'duelingDQN' if params['dueling'] else 'DQN'
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_{model_type}"
    policy_dqn.save(f"{MODEL_PATH}/{out_name}.pt")


def run_eval_episode(env, ob_dim, policy):
    # Reset the environment for a new episode
    ob, _ = env.reset()

    episode_over = False
    episode_reward = 0
    
    while not episode_over:
        # Sample an action from the policy
        with torch.no_grad():
            ac = policy(encode_ob(ob, ob_dim, env)).argmax().item()

        # Run the action on the environment
        new_ob, reward, terminated, truncated, _ = env.step(ac)

        # Collect the reward for this step
        episode_reward += reward

        # Update the observation and go for the next step
        ob = new_ob

        # Check if the episode is over
        episode_over = terminated or truncated

    return episode_reward


def evaluate(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize and load the policy network
    print('\nLoading policy network...\n')
    if params['dueling']:
        policy = DuelingDQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    else:
        policy = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    model_type = 'DDQN' if params['ddqn'] else 'duelingDQN' if params['dueling'] else 'DQN'
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_{model_type}"
    policy.load(f"{MODEL_PATH}/{out_name}.pt")

    # Run episodes
    for e in range(params['episodes']):
        episode_reward = run_eval_episode(env, ob_dim, policy)
        print(f"Episode {e+1}: Reward = {episode_reward}")
                
    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FrozenLake-v1', help='Environment name')
    parser.add_argument('--is_slippery', action='store_true', help='Indicates if the transition probabilities are non-deterministic')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--val_episodes', type=int, default=10, help='Number of episodes in the validation process')
    parser.add_argument('--network_sync_rate', type=int, default=10, help='Frequency at which policy and target networks are synced')
    parser.add_argument('--max_buffer_size', type=int, default=10000, help='Maximum capacity of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of experiences sampled from the replay buffer for each training iteration')
    parser.add_argument('--df', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of each hidden layer')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--e', type=float, default=1.0, help='The starting value for epsilon')
    parser.add_argument('--e_min', type=float, default=0.0, help='The minimum value for epsilon')
    parser.add_argument('--e_decay_rate', type=float, default=0.0, help='Epsilon decay rate')
    parser.add_argument('--ddqn', action='store_true', help='Use double DQN')
    parser.add_argument('--dueling', action='store_true', help='Use dueling DQN')
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