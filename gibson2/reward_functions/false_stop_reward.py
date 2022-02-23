from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class FalseStopReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(FalseStopReward, self).__init__(config)
        self.false_stop_reward = self.config.get(
            'false_stop_reward', 0.0
        )
        self.require_stop = self.config.get('REQUIRE_ACTIVE_STOP', False)
        self.dist_tol = self.config.get('dist_tol', 0.5)
        print('false_stop_reward', self.false_stop_reward)
        print('require_stop', self.require_stop)

    def get_reward(self, task, env, action):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        game_should_over = l2_distance(
            env.robots[0].get_position()[:2],
            task.target_pos[:2]) < self.dist_tol
        success = game_should_over and (action is None)
        if self.require_stop and (not game_should_over) and (action is None):
            reward = self.false_stop_reward
        else:
            reward = 0.0
        # reward = self.success_reward if success else 0.0
        return reward

    def update_weights(self, new_weights):
        self.false_stop_reward = new_weights.get(
            'false_stop_reward', self.false_stop_reward
        )
