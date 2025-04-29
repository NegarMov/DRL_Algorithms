<h1 align="center">Deep Reinforcement Learning Algorithms</h1>

![Python 3](https://img.shields.io/badge/Python-3.11-blue.svg)
![Pytorch](https://img.shields.io/badge/Pytorch-2.6.0-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains implementations of various deep reinforcement learning algorithms, primarily based on the content covered in the UC Berkeley [Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) course. 
This repository was created as part of my self-study to gain a deeper understanding of these algorithms.

All models are implemented in PyTorch and are trained on Gymnasium environments.

## Algorithms

Here are a list of different algorithms implemented in this repository:

<details>
  <summary><h3>Imitation Learning</h3></summary>
  
  Behavioural Cloning and DAgger algorithms were implemented in this part. The code for this part of the project is largely adapted from UC Berkeley [Homework 1](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw1).
  TODO parts from the original homework have been completed, some modifications were made for simplicity, and the code was updated to work with Gymnasium instead of Gym.
  
  **Results** \
  In the Half Cheetah task, Behavioural Cloning performs pretty well, achieving nearly 80% of the expert’s performance.
  However, in the other tasks —particularly Walker2d, which seems to be more complicated than the others— it performs very poorly.
  In general, it is evident that DAgger outperforms BC in all tasks by addressing the distributional shift problem.

  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Imitation%20Learning.jpg" alt="BC vs. DAgger" width="600"/>
  </p>
    
</details>

<details>
  <summary><h3>Policy Gradient</h3></summary>
  
  For this part, REINFORCE algorithm was implemented. 
  This algorithm works for both discrete and continuous action spaces. 
  For discrete action spaces, the model learns to represent a categorical distribution and for continuous action spaces, the model learns to represent a normal distribution over actions.
  
  **Results** \
  Training reward plot on HalfCheetah environment:

  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/HalfCheetah_PG.png" alt="Policy Gradient Training Rewards" width="300"/>
  </p>
  
  Policy Gradient in action on HalfCheetah environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/HalfCheetah_PG.gif" alt="Policy Gradient HalfCheetah Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>Actor-Critic</h3></summary>
  
  Implementation of an Actor-Critic method using Generalized Advantage Estimation (GAE) to reduce variance in the policy gradient estimates.
  
  We can efficiently implement the generalized advantage estimator by recursively computing:

  <p align="center">
    $A^\pi(s_t, a_t)\approx\delta_t=r(s_t, a_t)+\gamma V_\phi^\pi(s_{t+1})-V_\phi^\pi(s_{t})$
    <br>
    $A_{GAE}^\pi(s_t, a_t)=\delta_t+\gamma\lambda A_{GAE}^\pi(s_{t+1}, a_{t+1})$
  </p>
  
  **Results** \
  Training reward plot on HalfCheetah environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/HalfCheetah_Actor-Critic.png" alt="Actor-Critic Training Rewards" width="300"/>
  </p>
  
  Actor-Critic in action on HalfCheetah environment:
    
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/HalfCheetah_Actor-Critic.gif" alt="Actor-Critic HalfCheetah Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>Tabular Q-Learning</h3></summary>
  
  This algorithm uses a Q-table to store action values, hence is only suitable for environment with small discrete state spaces.
  During the training, actions are chosen using an epsilon-greedy policy for exploration-exploitation balance.
  Additionally, learning rate decay is adopted to achieve a more stable convergence.
  
  **Results** \
  Training reward plots on FrozenLake (slippery and non-slippery environments):

  <table align="center">
      <tr>
          <th>Slippery</th>
          <th>Non-slippery</th>
      </tr>
      <tr>
          <td>
              <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/FrozenLake_slippery_Q%20Learning.png?raw=true" alt="Slippery" width="350">
          </td>
          <td>
              <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/FrozenLake_Q%20Learning.png?raw=true" alt="Non-slippery" width="350">
          </td>
      </tr>
  </table>

</details>

<details>
  <summary><h3>Vanilla DQN, Double DQN (DDQN), and Dueling (Double) DQN</h3></summary>
  
  In this part, 3 variants of the DQN algorithm are implemented and a target network is used in all the algorithms to stabilize learning. 
  
  DDQN improves DQN by decoupling the action selection from the action evaluation, hence reducing the potential for overestimation.
  
  Dueling DQN splits the Q-values in two different parts, the value function V(s) and the advantage function A(s, a):
  <p align="center">
    $Q(s, a) = V(s) + A(s, a)$
  </p>
  To achieve this, the same neural network splits its last layer in two parts, one of them to estimate V(s) and the other one to estimate A(s, a).
  However, the problem with this approach is that given the function Q, we cannot determine the values of V and A.
  To address this issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action<sup>1</sup>.
  <p align="center">
    $Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|}\sum_{a'}{A(s, a')})$
  </p>
  
  **Results** \
  Comparison of training rewards for DQN, DDQN, and Dueling (Double) DQN on the CartPole environment:
  
  <table align="center">
      <tr>
          <th>Vanilla DQN</th>
          <th>DDQN</th>
          <th>Dueling DQN</th>
      </tr>
      <tr>
          <td>
              <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Cartpole_DQN.png" alt="DQN" width="350">
          </td>
          <td>
              <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Cartpole_Double DQN.png" alt="DDQN" width="350">
          </td>
          <td>
              <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Cartpole_Dueling DQN.png" alt="Dueling DQN" width="350">
          </td>
      </tr>
  </table>

  Please note that the best-performing model (based on training and evaluation runs) is selected and saved as the final model (indicated with a dashed red line).
  
  Dueling DQN in action on CartPole environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Cartpole_Dueling DQN.gif" alt="Dueling DQN CartPole Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>Deep Deterministic Policy Gradient (DDPG)</h3></summary>
  
  One of the main drawbacks of DQN is its inability to run on environments with a continious action space by nature.
  The main problem lies in getting the argmax over actions to find the optimal action, which is not feasable in continious action spaces.
  DDPG addresses this by learning a policy $`\mu(s)`$ in a way that:
  <p align="center">
    $max_a Q(s, a) \approx Q(s, \mu(s))$
  </p>
  In this algorithm, both actor and critic networks use target networks. These networks are synced with the main network using Polyak soft update.
  Also Action exploration is achieved by adding random normal noise to the actions.
  Additionally, to prevent overfitting, warmup episodes are run before the start of training and dropout is used in the critic network.
  
  This implementation is based on the pseudocode described in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
  
  **Results** \
  Training reward plot on Pendulum environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Pendulum_DDPG.png" alt="DDPG Training Rewards" width="300"/>
  </p>
  
  DDPG in action on Pendulum environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Pendulum_DDPG.gif" alt="DDPG Pendulum Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>CNN-DQN on Atari Games</h3></summary>
  
  This algorithm combines the dueling DQN method from the last part with convolutional neural networks to solve Atari environments.
  The dueling model architecture and hyperparameters are adapted from [bhctsntrk/OpenAIPong-DQN](https://github.com/bhctsntrk/OpenAIPong-DQN).
  
  **Preprocessing**
  To decrease computational cost and prepare the frames to serve as the input of the model, various preprocessing steps are applied using Gymnasium wrappers.
  These preprocessing steps are similar to that of the paper "Playing Atari with Deep Reinforcement Learning"<sup>2</sup>. Specifically:
  - Frames are converted to grayscale.
  - Frames are resized and cropped to reducing computational cost and focus on the playing area.
  - Every 4th frame is skipped, and each input to the Q-function consists of 4 stacked frames to address partial observability of the environment.
  Also, 3 additional game-specific wrappers can be used in the implemented code, however they are all disabled for the Pong game.
  - Clip reward: Clips the reward to the (-1, 1).
  - Episodic life: Ends an episode on life loss.
  - Fire reset: Some of the games (such as Breakout) require the player to press 'Fire' for the game to start. This wrapper performs the 'Fire' action upon each environment reset.
  
  **Results** \
  Training reward plot on Pong game:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Pong_CNN-DQN.png" alt="CNN-DQN Training Rewards" width="500"/>
  </p>
  
  CNN-DQN in action on Pong game:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Pong_CNN-DQN.gif" alt="CNN-DQN Pong Video" width="300"/>
  </p>
    
</details>

## Usage
Training and evaluation scripts are provided for each algorithm as a batch file under the `scripts` directory. 

Also saved model parameters can be found in `_models`. By default, the test scripts use this directory to load the model.

## References
1. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
2. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
