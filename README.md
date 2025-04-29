<h1 align="center">Deep Reinforcement Learning Algorithms</h1>

![Python 3](https://img.shields.io/badge/Python-3.11-blue.svg)
![Pytorch](https://img.shields.io/badge/Pytorch-2.6.0-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains implementations of various deep reinforcement learning algorithms, primarily based on the content covered in the UC Berkeley [Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) course. 
This repository was created as part of my self-study to gain a deeper understanding of these algorithms.

All models are implemented in PyTorch and are trained on Gymnasium environments.

## Algorithms

Here is a list of different algorithms implemented in this repository:

<details>
  <summary><h3>Imitation Learning</h3></summary>
  
  Behavioral Cloning (BC) and DAgger algorithms are implemented in this section. The code for this part of the project is largely adapted from UC Berkeley [Homework 1](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw1).
  TODO sections from the original homework have been completed, some modifications have been made for simplicity, and the code has been updated to work with Gymnasium instead of Gym.
  
  **Results** \
  In the Half Cheetah task, Behavioral Cloning performs quite well, achieving nearly 80% of the expert’s performance. 
  However, in other tasks —particularly Walker2d, which seems to be more complicated than the others— it performs very poorly. 
  In general, it is evident that DAgger outperforms BC in all tasks by addressing the distributional shift problem.

  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Imitation%20Learning.jpg" alt="BC vs. DAgger" width="600"/>
  </p>
    
</details>

<details>
  <summary><h3>Policy Gradient</h3></summary>

  In this section, the REINFORCE algorithm is implemented. 
  This algorithm works for both discrete and continuous action spaces. 
  For discrete action spaces, the model learns to represent a categorical distribution, while for continuous action spaces, it learns to represent a normal distribution over actions.
  
  **Results** \
  Training reward plot for the HalfCheetah environment:

  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/HalfCheetah_PG.png" alt="Policy Gradient Training Rewards" width="300"/>
  </p>
  
  Policy Gradient in action on the HalfCheetah environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/HalfCheetah_PG.gif" alt="Policy Gradient HalfCheetah Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>Actor-Critic</h3></summary>
  
  This section implements an Actor-Critic method and uses Generalized Advantage Estimation (GAE) to reduce variance in the policy gradient estimates.
  
  We can efficiently implement the generalized advantage estimator by recursively computing:

  <p align="center">
    $A^\pi(s_t, a_t)\approx\delta_t=r(s_t, a_t)+\gamma V_\phi^\pi(s_{t+1})-V_\phi^\pi(s_{t})$
    <br>
    $A_{GAE}^\pi(s_t, a_t)=\delta_t+\gamma\lambda A_{GAE}^\pi(s_{t+1}, a_{t+1})$
  </p>
  
  **Results** \
  Training reward plot for the HalfCheetah environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/HalfCheetah_Actor-Critic.png" alt="Actor-Critic Training Rewards" width="300"/>
  </p>
  
  Actor-Critic in action on the HalfCheetah environment:
    
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/HalfCheetah_Actor-Critic.gif" alt="Actor-Critic HalfCheetah Video" width="250"/>
  </p>
    
</details>

<details>
  <summary><h3>Tabular Q-Learning</h3></summary>
  
  This algorithm uses a Q-table to store action values, making it suitable only for environments with small discrete state spaces. 
  During training, actions are chosen using an epsilon-greedy policy to balance exploration and exploitation. 
  Additionally, learning rate decay is employed to achieve more stable convergence.
  
  **Results** \
  Training reward plots for the FrozenLake environment:

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
  
  In this section, three variants of the DQN algorithm are implemented and a target network is utilized in all cases to stabilize learning. 
  
  **DDQN** improves DQN by decoupling action selection from action evaluation, hence reducing the potential for overestimation.
  
  **Dueling DQN** splits the Q-values in two different components, the value function V(s) and the advantage function A(s, a):
  <p align="center">
    $Q(s, a) = V(s) + A(s, a)$
  </p>
  To achieve this, the same neural network splits its last layer in two parts, one of them to estimate V(s) and the other one to estimate A(s, a).
  However, the problem posed by this approach is that given the function Q, we cannot uniquely determine the values of V and A.
  To address this issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action<sup>1</sup>.
  <p align="center">
    $Q(s, a) = V(s) + (A(s, a) - max_{a'}{A(s, a')})$
  </p>
  Or alternatively:
  <p align="center">
    $Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|}\sum_{a'}{A(s, a')})$
  </p>
  
  **Results** \
  The following table compares training rewards for Vanilla DQN, DDQN, and Dueling DQN on the CartPole environment:
  
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

  Please note that the best-performing model (based on training and evaluation runs) is selected and saved as the final model (indicated by a dashed red line).
  
  Dueling DQN in action on the CartPole environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Cartpole_Dueling DQN.gif" alt="Dueling DQN CartPole Video" width="300"/>
  </p>
    
</details>

<details>
  <summary><h3>Deep Deterministic Policy Gradient (DDPG)</h3></summary>
  
  One of the main drawbacks of DQN is its inability to operate in environments with a continuous action space. 
  The primary challenge lies in obtaining the argmax over actions to find the optimal action, which is not feasible in continuous action spaces. 
  DDPG addresses this by learning a policy $`\mu(s)`$ such that:
  <p align="center">
    $max_a Q(s, a) \approx Q(s, \mu(s))$
  </p>
  In this implementation, both actor and critic networks utilize target networks. These networks are synchronized with the main network using Polyak soft updates. 
  Also, action exploration is achieved by adding random normal noise to the actions. 
  Additionally, to prevent overfitting, warmup episodes are executed before the start of training, and dropout is employed in the critic network.
  
  This implementation is based on the pseudocode described in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
  
  **Results** \
  Training reward plot on the Pendulum environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Pendulum_DDPG.png" alt="DDPG Training Rewards" width="300"/>
  </p>
  
  DDPG in action on the Pendulum environment:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Pendulum_DDPG.gif" alt="DDPG Pendulum Video" width="300"/>
  </p>
    
</details>

<details>
  <summary><h3>CNN-DQN on Atari Games</h3></summary>
  
  This algorithm combines the dueling DQN method from the previous section with convolutional neural networks to solve Atari environments. 
  The dueling model architecture and hyperparameters are adapted from [bhctsntrk/OpenAIPong-DQN](https://github.com/bhctsntrk/OpenAIPong-DQN).
  
  **Preprocessing**
  To decrease computational cost and prepare the frames for input to the model, various preprocessing steps are applied using Gymnasium wrappers. 
  These preprocessing steps are similar to those described in "Playing Atari with Deep Reinforcement Learning"<sup>2</sup>. Specifically:
  - Frames are converted to grayscale.
  - Frames are resized and cropped to reducing computational cost and focus on the playing area.
  - Every 4th frame is skipped, and each input to the Q-function consists of 4 stacked frames to address partial observability of the environment.
  
  Additionally, 3 game-specific wrappers can be utilized in the implemented code; however, they are all disabled for the Pong game:
  - Clip reward: Clips the rewards to (-1, 1).
  - Episodic life: Ends an episode on life loss.
  - Fire reset: Some games (such as Breakout) require the player to press 'Fire' for the game to start. This wrapper performs the 'Fire' action on each environment reset.
  
  **Results** \
  Training reward plot on the Pong game:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/graphs/Pong_CNN-DQN.png" alt="CNN-DQN Training Rewards" width="500"/>
  </p>
  
  CNN-DQN in action on the Pong game:
  
  <p align="center">
    <img src="https://github.com/NegarMov/DRL_Algorithms/blob/master/_assets/videos/Pong_CNN-DQN.gif" alt="CNN-DQN Pong Video" width="300"/>
  </p>
    
</details>

## Usage
Training and evaluation scripts for each algorithm are provided as batch files in the `scripts` directory.

Additionally, saved model parameters can be found in the  `_models` directory. By default, the test scripts will load the model from this directory.

## References
1. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
2. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
