import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

import torch


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

    return env, ob_dim, ac_dim


def rescale_action(ac, lower_bound, higher_bound, noise=None):
    if noise:
        noisy_ac = np.clip(ac + noise, -1, 1)
        scaled_ac = lower_bound + ((noisy_ac + 1.0) / 2) * (higher_bound - lower_bound)
        return noisy_ac, scaled_ac

    scaled_ac = lower_bound + ((ac + 1.0) / 2) * (higher_bound - lower_bound)
    return scaled_ac


def train(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(params['max_buffer_size'])

    # Initialize the networks and copy the weights to target network
    actor = Actor(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['actor_lr'])
    target_actor = Actor(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    target_actor.load_state_dict(actor.state_dict())

    critic = Critic(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'], params['critic_lr'])
    target_critic = Critic(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    target_critic.load_state_dict(critic.state_dict())
    
    # Array to store aggregated rewards
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])
    val_rewards = np.zeros(params['episodes']//params['log_freq'])

    # Run episodes
    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset(seed=params['seed'])

        episode_over = False

        ###################
        ### COLLECT A PATH
        ###################

        while not episode_over:
            # Select the action
            with torch.no_grad():
                ac = actor(ob).numpy()

            # Add noise to the action and rescale it
            noise = params['noise_scale'] * np.random.randn(ac_dim)
            noisy_ac, scaled_ac = rescale_action(ac, env.action_space.low, env.action_space.high, noise)

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(scaled_ac)

            # Add the transition to raplay buffer
            replay_buffer.append((ob, noisy_ac, new_ob, reward, terminated)) 

            # Collect the reward for this step
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation
            ob = new_ob

            # Check if the episode is over
            episode_over = terminated or truncated

        ###################
        ### DQN UPDATE
        ###################

        if len(replay_buffer) > params['batch_size'] and e > params['warmup_episodes']:
            obs, acs, next_obs, rewards, terminateds = replay_buffer.sample(params['batch_size'])

            # Critic update
            with torch.no_grad():
                target_acs = target_actor(next_obs)
                target_q = target_critic(next_obs, target_acs)
                target_q = rewards + (1 - terminateds) * params['df'] * target_q

            current_q = critic(obs, acs)
                    
            critic_loss = critic.loss_fn(current_q, target_q)
            
            critic.optimizer.zero_grad()
            critic_loss.backward()
            critic.optimizer.step()

            # Actor update
            policy_acs = actor(obs)

            actor_loss = -critic(obs, policy_acs).mean()
            
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            # Perform Polyak soft update
            for t_param, param in zip(target_actor.parameters(), actor.parameters()):
                t_param.data.copy_(params['tau'] * t_param.data + (1 - params['tau']) * param.data)
            
            for t_param, param in zip(target_critic.parameters(), critic.parameters()):
                t_param.data.copy_(params['tau'] * t_param.data + (1 - params['tau']) * param.data)

        ###################
        ### LOG METRICS
        ###################

        if (e + 1) % params['log_freq'] == 0:
            # Run validation 
            for _ in range(params['val_episodes']):
                val_rewards[e//params['log_freq']] += run_eval_episode(env, actor)

            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            val_rewards[e//params['log_freq']] /= params['val_episodes']
            print(f"\nEpisode {e+1}{' (warmup)' if e <= params['warmup_episodes'] else ''}: Training reward = {reward_per_episodes[e//params['log_freq']]}, Validation reward: {val_rewards[e//params['log_freq']]}")         

    env.close()

    # Plot the results
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes, label="Training Reward")
    plt.plot(np.arange(0, params['episodes'], params['log_freq']), val_rewards, label="Validation Reward")
    plt.xlabel('Episode Number')
    plt.ylabel(f"Average Reward per {params['log_freq']} Episodes")
    plt.legend()
    plt.show()

    # Save the trained networks
    out_name = f"{params['env_name']}_DDPG"
    
    print('\nSaving actor parameters...')
    actor.save(f"{MODEL_PATH}/{out_name}.pt")


def run_eval_episode(env, policy):
    # Reset the environment for a new episode
    ob, _ = env.reset()

    episode_over = False
    episode_reward = 0
    
    while not episode_over:
        # Select the action
        with torch.no_grad():
            ac = policy(ob).numpy()
        
        # Rescale the action
        ac = rescale_action(ac, env.action_space.low, env.action_space.high)

        # Run the action on the environment
        new_ob, reward, terminated, truncated, _ = env.step(ac)

        # Collect the reward for this step
        episode_reward += reward

        # Update the observation
        ob = new_ob

        # Check if the episode is over
        episode_over = terminated or truncated

    return episode_reward


def evaluate(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim = create_env(params)

    # Initialize and load the policy network
    print('\nLoading policy network...\n')
    policy = Actor(ob_dim, ac_dim, params['hidden_layers'], params['hidden_size'])
    out_name = f"{params['env_name']}_DDPG"
    policy.load(f"{MODEL_PATH}/{out_name}.pt")

    # Run episodes
    for e in range(params['episodes']):
        print(f"Episode {e+1}: Reward = {run_eval_episode(env, policy)}")
                
    env.close()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--warmup_episodes', type=int, default=100, help='Number of episodes before training start')
    parser.add_argument('--val_episodes', type=int, default=10, help='Number of episodes in the validation process')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--max_buffer_size', type=int, default=int(1e6), help='Maximum capacity of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of experiences sampled from the replay buffer for each training iteration')
    parser.add_argument('--df', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of each hidden layer')
    parser.add_argument('--actor_lr', type=float, default=1e-2, help='Learning rate for the actor network')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='Learning rate for the critic network')
    parser.add_argument('--tau', type=float, default=0.99, help='Polyak averaging coefficient')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Action noise scale')
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