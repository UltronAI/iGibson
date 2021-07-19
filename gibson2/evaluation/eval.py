from gibson2.utils.utils import parse_config
import numpy as np
import json
import os
import sys
from gibson2.envs.igibson_env import iGibsonEnv
import logging
logging.getLogger().setLevel(logging.WARNING)

from onpolicy_agent import rMAPPOAgent
import argparse


class Evaluator:
    def __init__(self, config_file, split, episode_dir, eval_episodes_per_scene=100):
        self.config_file = config_file #os.environ['CONFIG_FILE']
        self.split = split #os.environ['SPLIT']
        self.episode_dir = episode_dir #os.environ['EPISODE_DIR']
        self.eval_episodes_per_scene = eval_episodes_per_scene 
            #os.environ.get('EVAL_EPISODES_PER_SCENE', 100)

    def eval(self, agent):
        env_config = parse_config(self.config_file)

        task = env_config['task']
        if task == 'point_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'episode_return']}
        elif task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return']}
        else:
            assert False, 'unknown task: {}'.format(task)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        idx = 0
        for json_file in os.listdir(split_dir):
            scene_id = json_file.split('.')[0]
            json_file = os.path.join(split_dir, json_file)

            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 2.0,
                             physics_timestep=1.0 / 40.0)

            for _ in range(num_episodes_per_scene):
                idx += 1
                print('Episode: {}/{}'.format(idx, total_num_episodes))
                try:
                    agent.reset()
                except:
                    pass
                state = env.reset()
                episode_return = 0.0
                while True:
                    action = env.action_space.sample()
                    action = agent.act(state)
                    state, reward, done, info = env.step(action)
                    episode_return += reward
                    if done:
                        break

                metrics['episode_return'] += episode_return
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]

        for key in metrics:
            metrics[key] /= total_num_episodes
            print('Avg {}: {}'.format(key, metrics[key]))


def main(args):
    parser = argparse.ArgumentParser(
        description='onpolicy-eval', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, default=None)
    args = parser.parse_known_args(args)[0]

    config_file = "../examples/configs/locobot_point_nav_discrete.yaml"
    split = "eval"
    episodes_dir = "../data/episodes_data/point_nav"
    evaluator = Evaluator(config_file, split, episodes_dir)
    agent = rMAPPOAgent(args.model)

    evaluator.eval(agent)

if __name__ == "__main__":
    main(sys.argv[1:])