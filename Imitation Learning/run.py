import pickle
import time
import gymnasium as gym

import numpy as np
import torch

import infrastructure.pytorch_util as ptu
from infrastructure import utils
from infrastructure.replay_buffer import ReplayBuffer
from policy.MLP_policy import MLPPolicySL
from policy.loaded_gaussian_policy import LoadedGaussianPolicy


MAX_VIDEO_LEN = 40  # we overwrite this in the code below

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    #############
    ## ENV
    #############

    # Make the gym environment
    env = gym.make(params['env_name'], render_mode='rgb_array')
    env.reset(seed=seed)

    # Maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if 'model' in dir(env):
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT
    #############

    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # replay buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy...')

    #######################
    ## TRAINING LOOP
    #######################

    # init vars at beginning of training
    total_envsteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        print("\nCollecting data to be used for training...")
        if itr == 0:
            # BC training from expert data.
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            paths, envsteps_this_batch = utils.sample_trajectories(
              env,
              actor, 
              params['batch_size'], 
              params['ep_len']
            )

            # relabel the collected obs with actions from a provided expert policy
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                for path in paths:
                  path['action'] = expert_policy.get_action(path["observation"])

        total_envsteps += envsteps_this_batch
        # add collected data to replay buffer
        replay_buffer.add_rollouts(paths)

        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):

          # for imitation learning, we only need observations and actions.  
          selected_indices = np.random.choice(len(replay_buffer), size=params['train_batch_size'], replace=False)
          ob_batch, ac_batch = replay_buffer.obs[selected_indices], replay_buffer.acs[selected_indices]

          # use the sampled data to train an agent
          train_log = actor.update(ob_batch, ac_batch)
          training_logs.append(train_log)

        # save eval metrics
        print("\nCollecting data for eval...")
        eval_paths, _ = utils.sample_trajectories(
            env, actor, params['eval_batch_size'], params['ep_len'])

        logs = utils.compute_metrics(paths, eval_paths)
        # compute additional metrics
        logs.update(training_logs[-1]) # Only use the last log for now
        logs["Train_EnvstepsSoFar"] = total_envsteps
        logs["TimeSinceStart"] = time.time() - start_time
        if itr == 0:
            logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))

    # save eval rollouts as a video
    print('\nCollecting video rollouts eval')
    eval_video_paths = utils.sample_n_trajectories(env, actor, 1, MAX_VIDEO_LEN, True)
    if eval_video_paths is not None:
        utils.display_paths_as_videos(
            eval_video_paths,
            max_videos_to_save=1,
            fps=fps,
            video_title='eval_rollouts')

    # save the model parameters if specified so
    if params['save_params']:
        print('\nSaving agent params')
        actor.save(params['model_save_path'])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=1000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int, default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int, default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    if args.do_dagger:
        assert args.n_iter > 1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        assert args.n_iter == 1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    if args.save_params:
        assert args.model_save_path is not None, ('Specify a path for saving the model params')

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()