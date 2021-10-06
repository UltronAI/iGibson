from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance

class PersonalSpaceViolationReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(PersonalSpaceViolationReward, self).__init__(config)
        self.personal_space_violation_weight = self.config.get(
            'personal_space_violation_weight', 0.0)
        self.personal_space_violation_threshold = self.config.get(
            'personal_space_violation_threshold', 1.5)
        print('personal_space_violation_threshold', self.personal_space_violation_threshold)

    def get_reward(self, task, env):
        """
        Reward is self.personal_space_violation_weight if there is personal_space_violation
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        violation_count = 0
        robot_pos = env.robots[0].get_position()[:2]
        for ped in self.pedestrians:
            ped_pos = ped.get_position()[:2]
            _, geo_dist = env.scene.get_shortest_path(
                    env.task.floor_num,
                    robot_pos[:2], ped_pos[:2], entire_path=False)
            if geo_dist < self.personal_space_violation_threshold:
                violation_count += 1
        return violation_count * self.personal_space_violation_weight
