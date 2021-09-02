import numpy as np
from functools import reduce
import gibson2
from gibson2.envs.igibson_env import iGibsonEnv as inner_iGibsonEnv

import gym


class iGibsonEnv(object):

    def __init__(self, args, scene_id, device_idx=None):
        self.num_agents = args.num_agents # TODO: remove multi-agent?
        self.mode = args.mode
        self.scenario_name = args.scenario_name
        self.config = gibson2.__path__[0] + '/examples/configs/' + str(self.scenario_name) + '.yaml'

        try:
            reward_weights = {
                'collision_reward_weight': args.collision_reward_weight,
                'potential_reward_weight': args.potential_reward_weight
            }
        except:
            reward_weights = None

        try:
            image_size = (args.frame_width, args.frame_height)
        except:
            image_size = None

        self.env = inner_iGibsonEnv(config_file=self.config,
                                    mode=self.mode,
                                    scene_id=scene_id,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 240.0,
                                    device_idx=args.render_gpu_ids[0] if device_idx is None else device_idx,
                                    reward_weights=reward_weights,
                                    image_size=image_size
                                    )
        
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        for agent_id in range(self.num_agents):
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
            self.action_space.append(self.env.action_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        for key, value in obs.items():
            obs[key] = [value]
        return obs, None

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])

        for key, value in obs.items():
            obs[key] = [value]
        return obs, [[reward]], [done], [info]
 
    def close(self):
        self.env.close()


if __name__ == '__main__':
    # check 15 scenes
    scenes = ["Ihlen_1_int",
            "Rs_int",
            "Pomaria_1_int",
            "Merom_1_int",
            "Pomaria_2_int",
            "Pomaria_0_int",
            "Wainscott_1_int",
            "Benevolence_1_int",
            "Ihlen_0_int",
            "Wainscott_0_int",
            "Beechwood_0_int",
            "Beechwood_1_int",
            "Merom_0_int",
            "Benevolence_2_int",
            "Benevolence_0_int"]

    for scene in scenes:
        gpu_id = 0
        # print(scene)
        _env = iGibsonEnv(scene, 1)
        print(scene)
        _env.close()