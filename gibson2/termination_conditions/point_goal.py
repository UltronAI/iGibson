from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.utils.utils import l2_distance


class PointGoal(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.require_stop = self.config.get('REQUIRE_ACTIVE_STOP', False)

    def get_termination(self, task, env, action):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        task_done = l2_distance(
            env.robots[0].get_position()[:2],
            task.target_pos[:2]) < self.dist_tol
        if self.require_stop:
            done = action is None
            success = task_done and (action is None)
        else:
            done = task_done
            success = task_done
        return done, success
