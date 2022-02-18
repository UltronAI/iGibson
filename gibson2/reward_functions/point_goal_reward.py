from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class PointGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(PointGoalReward, self).__init__(config)
        self.success_reward = self.config.get(
            'success_reward', 10.0
        )
        self.false_stop_reward = self.config.get(
            'false_stop_reward', 0.0
        )
        self.dist_tol = self.config.get('dist_tol', 0.5)
        print('success_reward', self.success_reward)

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
        if not game_should_over and (action is None):
            reward = self.false_stop_reward
        elif success:
            reward = self.success_reward
        else:
            reward = 0.0
        # reward = self.success_reward if success else 0.0
        return reward

    def update_weights(self, new_weights):
        self.success_reward = new_weights.get(
            'success_reward', self.success_reward
        )
