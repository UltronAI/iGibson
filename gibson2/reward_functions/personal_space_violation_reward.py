from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance
import numpy as np

class PersonalSpaceViolationReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(PersonalSpaceViolationReward, self).__init__(config)
        self.personal_space_violation_reward = self.config.get(
            'personal_space_violation_reward', 0.0)
        self.personal_space_violation_threshold = self.config.get(
            'personal_space_violation_threshold', 1.5)
        if (isinstance(self.personal_space_violation_threshold, list) \
            or isinstance(self.personal_space_violation_threshold, tuple)
        ):
            self.personal_space_violation_threshold = max(self.personal_space_violation_threshold)
        self.use_increasing_violation_reward = bool(self.config.get(
            'use_increasing_violation_reward', False))
        self.use_updated_violation_reward = bool(self.config.get(
            'use_updated_violation_reward', False))
        print('personal_space_violation_reward', self.personal_space_violation_reward)
        print('use_increasing_violation_reward', self.use_increasing_violation_reward)
        print('use_updated_violation_reward', self.use_updated_violation_reward)

    def get_reward(self, task, env, action):
        """
        Reward is self.personal_space_violation_weight if there is personal_space_violation
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        violation_count = 0
        robot_pos = env.robots[0].get_position()[:2]
        for ped in task.pedestrians:
            ped_pos = ped.get_position()[:2]
            # _, geo_dist = env.scene.get_shortest_path(
            #         task.floor_num,
            #         np.array(robot_pos[:2]), np.array(ped_pos[:2]), entire_path=False)
            # if geo_dist < self.personal_space_violation_threshold:
            #     violation_count += 1
            #     break
            if self.use_updated_violation_reward:
                d = l2_distance(robot_pos, ped_pos)
                if d < self.personal_space_violation_threshold:
                    violation_count += self.personal_space_violation_threshold - d
            elif not self.use_increasing_violation_reward:
                if l2_distance(robot_pos, ped_pos) < self.personal_space_violation_threshold:
                    violation_count += 1
                    break
            else:
                d = l2_distance(robot_pos, ped_pos)
                if d < self.personal_space_violation_threshold:
                    violation_count += 2 - d / self.personal_space_violation_threshold
                    break
        return violation_count * self.personal_space_violation_reward

    def update_weights(self, new_weights):
        self.personal_space_violation_reward = new_weights.get(
            'personal_space_violation_reward', self.personal_space_violation_reward
        )
