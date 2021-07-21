import gibson2
import numpy as np
import os
import json

class EpisodeConfig:
    def __init__(self, scene_id, num_episodes, numpy_seed=0):
        np.random.seed(numpy_seed)
        self.scene_id = scene_id
        self.num_episodes = num_episodes
        self.numpy_seed = numpy_seed
        self.episode_index = -1
        self.episodes = [{} for _ in range(num_episodes)]

    def reset_episode(self):
        self.episode_index += 1
        if self.episode_index >= self.num_episodes:
            raise ValueError(
                "We have exhausted all {} episodes sampels".format(
                    self.num_episodes))

class PointNavEpisodesConfig(EpisodeConfig):
    def __init__(self, scene_id, num_episodes, numpy_seed=0):
        super(PointNavEpisodesConfig, self).__init__(
            scene_id, num_episodes, numpy_seed)
        
        # inital pos, goal pos, orientation
        for episode_index in range(num_episodes):
            self.episodes[episode_index] = {
                'initial_pos': None,
                'initial_orn': None,
                'target_pos': None
            }

    @classmethod
    def load_scene_episode_config(cls, path):
        with open(path) as f:
            config = json.load(f)
        episode_config = PointNavEpisodesConfig(
            scene_id=config['config']['scene_id'],
            num_episodes=config['config']['num_episodes'],
            numpy_seed=config['config']['numpy_seed']
        )
        episode_config.episodes = config['episodes']

        return episode_config