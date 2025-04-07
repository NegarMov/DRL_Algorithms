from collections import OrderedDict
import cv2
import numpy as np

import infrastructure.pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    
    # initialize env for the beginning of a new rollout
    ob, _ =  env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            img = env.render()
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        ac = policy.forward(ptu.from_numpy(ob))
        ac = ptu.to_numpy(ac)

        next_ob, rew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated
        
        # rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = 1 if done or steps == max_path_length else 0
        
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
    

########################################
########################################


def display_video(video_frames, name, fps=10):
    assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video display!"
    
    # Display each video individually
    for i in range(video_frames.shape[0]):
        video = video_frames[i]  # Get the i-th video
        for frame in video:
            # Convert frame to BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(name, frame_bgr)
            key = cv2.waitKey(int(1000 / fps))
            
            # Check if the window was closed by the user
            if key == ord('q') or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()  # Close the window after each video

def display_paths_as_videos(paths, max_videos_to_save=2, fps=10, video_title='video'):
    videos = [p['image_obs'] for p in paths]

    # Max rollout length
    max_videos_to_save = np.min([max_videos_to_save, len(videos)])
    max_length = videos[0].shape[0]
    for i in range(max_videos_to_save):
        if videos[i].shape[0] > max_length:
            max_length = videos[i].shape[0]

    # Pad rollouts to all be the same length
    for i in range(max_videos_to_save):
        if videos[i].shape[0] < max_length:
            padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
            videos[i] = np.concatenate([videos[i], padding], 0)

    # Display videos
    videos = np.stack(videos[:max_videos_to_save], 0)
    display_video(videos, video_title, fps=fps)
