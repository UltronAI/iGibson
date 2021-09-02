from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class SlackReward(BaseRewardFunction):
    """
    Slack reward
    """
    def __init__(self, config):
        super(SlackReward, self).__init__(config)
        self.slack_reward_weight = self.config.get(
            'slack_reward_weight', -0.01
        )

    def reset(self, task, env):
        pass

    def get_reward(self, task, env):
        return self.slack_reward_weight