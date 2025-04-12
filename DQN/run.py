import gymnasium as gym
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from DQN import DQN
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
    policy_dqn = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['lr'])
    target_dqn = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['lr'])

    target_dqn.load_state_dict(policy_dqn.state_dict())
    
    # Array to store aggregated rewards
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])

    update_count = 0

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset(seed=params['seed'])

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
                    ac = target_dqn(encode_ob(ob, ob_dim, env)).argmax().item()

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
            mini_batch = replay_buffer.sample(params['batch_size'])
            
            policy_q_values = []
            target_q_values = []

            for ob, ac, new_ob, reward, terminated in mini_batch:

                # Estimate the target Q values
                if terminated:
                    target = torch.FloatTensor([reward])
                else:
                    with torch.no_grad():
                        if params['ddqn']:
                            policy_ac = policy_dqn(encode_ob(new_ob, ob_dim, env)).argmax()
                            target = torch.FloatTensor(
                                reward + params['df'] * target_dqn(encode_ob(new_ob, ob_dim, env))[policy_ac]
                            )
                        else:
                            target = torch.FloatTensor(
                                reward + params['df'] * target_dqn(encode_ob(new_ob, ob_dim, env)).max()
                            )
                target_q = target_dqn(encode_ob(ob, ob_dim, env)) 
                
                # Adjust the specific action to the target that was just calculated
                target_q[ac] = target
                target_q_values.append(target_q)

                # Get the policy Q values
                policy_q_values.append(policy_dqn(encode_ob(ob, ob_dim, env)))
                    
            # Compute the loss and backpropagate
            policy_dqn.optimizer.zero_grad()

            loss = policy_dqn.loss_fn(torch.stack(policy_q_values), torch.stack(target_q_values))

            loss.backward()
            policy_dqn.optimizer.step()

            update_count += 1

            # Copy policy network weights to target network
            if update_count > params['network_sync_rate']:
                target_dqn.load_state_dict(policy_dqn.state_dict())

        if (e + 1) % params['log_freq'] == 0:
            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            print(f"\nEpisode {e+1}: Reward = {reward_per_episodes[e//params['log_freq']]}")

    env.close()

    # Plot the results
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes)
    plt.xlabel('Episode Number')
    plt.ylabel(f"Average Reward per {params['log_freq']} Episodes")
    plt.show()

    # Save the learned policy network
    print('\nSaving policy network...')
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_DQN"
    policy_dqn.save(f"{MODEL_PATH}/{out_name}.pt")


def evaluate(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize and load the policy network
    print('\nLoading policy network...\n')
    policy = DQN(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['lr'])
    out_name = f"{params['env_name']}{'_slippery' if params['is_slippery'] else ''}_DQN"
    policy.load(f"{MODEL_PATH}/{out_name}.pt")

    # Run episodes
    for e in range(params['episodes']):
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

        print(f"Episode {e+1}: Reward = {episode_reward}")
                
    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FrozenLake-v1', help='Environment name')
    parser.add_argument('--is_slippery', action='store_true', help='Indicates if the transition probabilities are non-deterministic')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
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