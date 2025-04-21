import gymnasium as gym

"""
    For environments where the user need to press FIRE for the game to start.
"""
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3


    def step(self, action):
        return self.env.step(action)


    def reset(self, seed=None):
        self.env.reset(seed=seed)

        new_ob, _, terminated, truncated, _ = self.env.step(1)
        
        if terminated or truncated:
            self.env.reset()
        
        return new_ob, _


"""
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
"""
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(EpisodicLifeEnv, self).__init__(env)
        
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False


    def step(self, action):
        new_ob, reward, terminated, truncated, _ = self.env.step(action)

        self.was_real_done = terminated or truncated

        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        
        self.lives = lives
        
        return new_ob, reward, terminated, truncated, _


    def reset(self, seed=None):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            new_ob, _ = self.env.reset(seed=seed)
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            new_ob, _, _, _, _ = self.env.step(0)
            self.was_real_reset = False
            
        self.lives = self.env.unwrapped.ale.lives()

        return new_ob