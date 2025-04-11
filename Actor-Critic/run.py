import gymnasium as gym
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from actor import Actor
from critic import Critic

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


def compute_advantage(episode_transitions, i, critic, params):
    return episode_transitions[i]['reward'] + \
        params['df'] * critic(episode_transitions[i]['new_ob']) - \
            critic(episode_transitions[i]['ob'])


def train(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim, discrete = create_env(params)

    # Initialize actor and critic networks
    actor = Actor(
        ob_dim, 
        ac_dim, 
        discrete, 
        hidden_layers=params['actor_hidden_layers'], 
        hidden_size=params['actor_hidden_size'], 
        lr=params['actor_lr']
    )

    critic = Critic(
        ob_dim,  
        hidden_layers=params['critic_hidden_layers'], 
        hidden_size=params['critic_hidden_size'], 
        lr=params['critic_lr']
    )
        
    # Arrays to store aggregated rewards and losses
    reward_per_episodes = np.zeros(params['episodes']//params['log_freq'])
    actor_loss_per_episodes = np.zeros(params['episodes']//params['log_freq'])
    critic_loss_per_episodes = np.zeros(params['episodes']//params['log_freq'])

    for e in tqdm(range(params['episodes'])):
        # Reset the environment for a new episode
        ob, _ = env.reset()

        episode_over = False
        episode_probs = []
        episode_transitions = []
        steps = 0

        ###################
        ### RUN AN EPISODE
        ###################
        
        while not episode_over:
            # Sample an action from the policy
            ac_dist = actor(ob)
            ac = ac_dist.sample()

            # Store action log probabilities for policy update
            episode_probs.append(ac_dist.log_prob(ac))
            
            # Convert the action to a numpy array
            ac = ac.data.cpu().numpy().astype(env.action_space.dtype)

            # Run the action on the environment
            new_ob, reward, terminated, truncated, _ = env.step(ac)

            # Collect the reward for this step
            episode_transitions.append({
                'reward': reward,
                'ob': ob,
                'new_ob': new_ob
            })
            reward_per_episodes[e//params['log_freq']] += reward

            # Update the observation and go for the next step
            ob = new_ob
            steps += 1

            # Check if the episode is over
            episode_over = terminated or truncated or steps > params['max_episode_len']

        ###################
        ### MODEL UPDATE
        ###################

        discounted_sum = 0
        # Compute the discounted rewards (used to train the critic)
        for i in reversed(range(steps)):
            episode_transitions[i]['discounted_reward'] = episode_transitions[i]['reward'] + params['df'] * discounted_sum
            discounted_sum = episode_transitions[i]['discounted_reward']

        # Compute the advantage of ith the step (used to train the actor)
        gae = 0
        for i in reversed(range(steps)):
            gae = compute_advantage(episode_transitions, i, critic, params) + \
                params['df'] * params['lambda'] * gae
            
            episode_transitions[i]['advantage'] = gae

        # Compute the loss and backpropagate
        actor.optimizer.zero_grad()
        critic.optimizer.zero_grad()

        # Actor loss
        episode_advantages = [t['advantage'] for t in episode_transitions]
        episode_advantages = torch.tensor(episode_advantages, dtype=torch.float32, device=actor.device)

        # Normalize the advantages to reduce variance
        episode_advantages = (episode_advantages - episode_advantages.mean()) / (episode_advantages.std() + 1e-7)
        
        episode_probs = torch.stack(episode_probs)
        if episode_probs.dim() > 1:
            episode_probs = episode_probs.sum(dim=-1)

        actor_loss = -1 * torch.sum(episode_probs * episode_advantages)

        # Critic loss
        critic_values = critic(np.array([t['ob'] for t in episode_transitions]))
        target_values = torch.tensor(np.array([t['discounted_reward'] for t in episode_transitions]), dtype=torch.float32, device=critic.device).unsqueeze(-1)
        critic_loss = critic.loss_fn(critic_values, target_values)

        actor_loss.backward()
        critic_loss.backward()

        actor.optimizer.step()
        critic.optimizer.step()

        # Collect the loss for this episode
        actor_loss_per_episodes[e//params['log_freq']] += actor_loss.item()
        critic_loss_per_episodes[e//params['log_freq']] += critic_loss.item()

        # Print progress at the end of each aggregation interval
        if (e + 1) % params['log_freq'] == 0:
            actor_loss_per_episodes[e//params['log_freq']] /= params['log_freq']
            critic_loss_per_episodes[e//params['log_freq']] /= params['log_freq']
            reward_per_episodes[e//params['log_freq']] /= params['log_freq'] 
            print(f"\nEpisode {e+1}: Actor loss = {actor_loss_per_episodes[e//params['log_freq']]}, Critic loss = {critic_loss_per_episodes[e//params['log_freq']]}, Reward = {reward_per_episodes[e//params['log_freq']]}")
                
    env.close()

    # Plot the results
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].plot(np.arange(0, params['episodes'], params['log_freq']), reward_per_episodes)
    axs[0].set_xlabel('Episode Number')
    axs[0].set_ylabel(f"Average Reward per {params['log_freq']} Episodes")

    axs[1].plot(np.arange(0, params['episodes'], params['log_freq']), actor_loss_per_episodes, label='Actor Loss')
    axs[1].plot(np.arange(0, params['episodes'], params['log_freq']), critic_loss_per_episodes, label='Critic Loss')
    axs[1].set_xlabel('Episode Number')
    axs[1].set_ylabel(f"Average loss per {params['log_freq']} Episodes")
    axs[1].legend()


    plt.tight_layout()
    plt.show()

    # Save the traied models
    print('\nSaving actor parameters...')
    actor.save(f"{MODEL_PATH}/{params['env_name']}_actor.pt")

    print('\nSaving critic parameters...')
    actor.save(f"{MODEL_PATH}/{params['env_name']}_critic.pt")


def evaluate(params):
    # Create the environment and get its properties
    env, ob_dim, ac_dim, discrete = create_env(params)

    # Initialize and load the policy
    policy = Actor(
        ob_dim, 
        ac_dim, 
        discrete, 
        hidden_layers=params['actor_hidden_layers'], 
        hidden_size=params['actor_hidden_size'], 
        lr=params['actor_lr']
    )
    print('\nLoading policy parameters...\n')
    policy.load(f"{MODEL_PATH}/{params['env_name']}_actor.pt")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_episode_len', type=int, default=1000, help='Maximum episode length')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency at which training rewards and losses are recorded (in episodes)')
    parser.add_argument('--df', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--actor_hidden_layers', type=int, default=2, help='Number of hidden layers of the actor network')
    parser.add_argument('--actor_hidden_size', type=int, default=64, help='Size of each hidden layer in the actor network')
    parser.add_argument('--actor_lr', type=float, default=1e-2, help='Learning rate of the actor network')
    parser.add_argument('--critic_hidden_layers', type=int, default=2, help='Number of hidden layers of the critic network')
    parser.add_argument('--critic_hidden_size', type=int, default=64, help='Size of each hidden layer in the critic network')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='Learning rate of the critic network')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    # Run the main function
    if params['eval']:
        evaluate(params)
    else:
        train(params)
