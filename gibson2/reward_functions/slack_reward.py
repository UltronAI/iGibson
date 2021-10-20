from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class SlackReward(BaseRewardFunction):
    """
    Slack reward
    """
    def __init__(self, config):
        super(SlackReward, self).__init__(config)
        self.slack_reward = self.config.get(
            'slack_reward', -0.01
        )
        print("slack reward", self.slack_reward)

    def reset(self, task, env):
        pass

    def get_reward(self, task, env):
        return self.slack_reward

    def update_weights(self, new_weights):
        self.slack_reward = new_weights.get(
            'slack_reward', self.slack_reward
        )