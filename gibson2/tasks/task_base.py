from abc import abstractmethod, ABCMeta


class BaseTask():
    """
    Base Task class.
    Task-specific reset_scene, reset_agent, get_task_obs, step methods are implemented in subclasses
    Subclasses are expected to populate self.reward_functions and self.termination_conditions
    """
    __metaclass__ = ABCMeta
    def __init__(self, env):
        self.config = env.config
        self.reward_functions = []
        self.termination_conditions = []
        self.task_obs_dim = self.config.get('task_obs_dim', 0)
        self.number_of_episodes = -1
        self.episode_index = -1

    @abstractmethod
    def reset_scene(self, env):
        """
        Task-specific scene reset

        :param env: environment instance
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        """
        Task-specific agent reset

        :param env: environment instance
        """
        raise NotImplementedError()

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        # reward = 0.0
        rewards = {}
        for reward_function in self.reward_functions:
            # reward += reward_function.get_reward(self, env)
            rewards[reward_function.name] = reward_function.get_reward(self, env, action)
        return rewards, info

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return done: whether the episode has terminated
        :return info: additional info
        """
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env, action)
            done = done or d
            success = success or s
            info[f'done_{condition.__class__.__name__}'] = d
        info['done'] = done
        info['success'] = success
        return done, info

    @abstractmethod
    def get_task_obs(self, env):
        """
        Get task-specific observation

        :param env: environment instance
        :return: task-specific observation (numpy array)
        """
        raise NotImplementedError()

    def step(self, env, info):
        """
        Perform task-specific step for every timestep

        :param env: environment instance
        """
        return info

    @abstractmethod
    def get_global_infos(self, env):
        """
        Get task-specific global infos

        :param env: environment instance
        :return: task-specific global infos (numpy array)
        """
        raise NotImplementedError()

    def update_reward_weights(self, new_weights):
        for reward in self.reward_functions:
            reward.update_weights(new_weights)
