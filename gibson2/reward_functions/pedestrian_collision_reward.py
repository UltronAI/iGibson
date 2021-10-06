from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance

class PedestrianCollisionReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(PedestrianCollisionReward, self).__init__(config)
        self.pedestrian_collision_reward = self.config.get(
            'pedestrian_collision_reward', 0.0)
        self.pedestrian_collision_threshold = self.config.get(
            'pedestrian_collision_threshold', 0.3)
        print('pedestrian_collision_reward', self.pedestrian_collision_reward)

    def get_reward(self, task, env):
        """
        Reward is self.pedestrian_collision_reward if there is collision with pedestrians
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        collision_count = 0
        robot_pos = env.robots[0].get_position()[:2]
        for ped in task.pedestrians:
            ped_pos = ped.get_position()[:2]
            if l2_distance(robot_pos, ped_pos) < self.pedestrian_collision_threshold:
                collision_count += 1
                break
        return collision_count * self.pedestrian_collision_reward
