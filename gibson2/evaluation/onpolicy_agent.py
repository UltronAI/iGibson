
from IPython import embed
import collections
from collections import OrderedDict
import os

import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

from onpolicy_config import get_config

from collections import defaultdict

import yaml

IMG_WIDTH = 320
IMG_HEIGHT = 180
TASK_OBS_DIM = 4

def _t2n(x):
    return x.detach().cpu().numpy()

def load_args(config_path):
    try:
        loader = yaml.CLoader
    except:
        loader = yaml.Loader

    configs = {}

    with open(config_path, 'rb') as f:
        yaml_data = yaml.load(f.read(), Loader=loader)
        for key, value in yaml_data.items():
            if not isinstance(value, dict):
                continue
            configs[key] = value['value']
            print(key)
            print(value['value'])
            print("----")

    return configs

class rMAPPOAgent:
    def __init__(self,
                 root_dir,
                 device=torch.device("cuda")
    ):
        self.model_dir = os.path.expanduser(root_dir)
        self.all_args = get_config().parse_known_args()[0]

        self.all_args.__dict__ = self.load_args(str(self.model_dir) + '/config.yaml')
        if 'multi_stage_mode' not in self.all_args.__dict__.keys():
            self.all_args.__dict__['multi_stage_mode'] = 0

        print("=" * 30)
        print(self.all_args.__dict__)
        print("=" * 30)
        self.all_args.use_pretrained_model = False
        self.all_args.use_multi_gpu = False

        self.load_observation_space()
        self.load_action_space()

        # modify args for igibson evaluation
        self.all_args.n_rollout_threads = 1
        self.all_args.num_agents = 1
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.num_agents = self.all_args.num_agents

        self.policy = Policy(self.all_args,
                             self.observation_space[0],
                             self.observation_space[0],
                             self.action_space[0],
                             device=device)

        self.restore(self.all_args.use_multi_gpu)

        self.policy.actor.eval()
        self.policy.critic.eval()

        self.buffer = SharedReplayBuffer(self.all_args,
                                         num_agents=1,
                                         obs_space=self.observation_space[0],
                                         share_obs_space=self.observation_space[0],
                                         act_space=self.action_space[0])

        self.eval_rnn_states = np.zeros((1, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        self.eval_masks = np.ones((1, 1, 1), dtype=np.float32)
        self.num_steps = 0

    def load_args(self, config_path):
        try:
            loader = yaml.CLoader
        except:
            loader = yaml.Loader

        configs = {}
        with open(config_path, 'rb') as f:
            yaml_data = yaml.load(f.read(), Loader=loader)
            for key, value in yaml_data.items():
                if not isinstance(value, dict):
                    continue
                configs[key] = value['value']

        return configs

    def reset(self):
        self.eval_rnn_states = np.zeros((1, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        self.eval_masks = np.ones((1, 1, 1), dtype=np.float32)
        self.num_steps = 0

    def act(self, original_obs):
        eval_obs = self.convert_obs(original_obs)

        concat_eval_obs = {}
        for key in eval_obs.keys():
            concat_eval_obs[key] = np.concatenate(eval_obs[key])
        eval_action, eval_rnn_states = self.policy.act(concat_eval_obs,
                                                       np.concatenate(self.eval_rnn_states),
                                                       np.concatenate(self.eval_masks),
                                                       deterministic=True)
        eval_actions = np.array(np.split(_t2n(eval_action), 1))
        self.eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 1))
        
        # if self.all_args.use_discrete_action:
        #     return self.recover_original_action(eval_actions[0][0])
        # else:
        return eval_actions[0][0]

    def process_state_dict(self, state_dict, use_multi_gpu):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'module' in key and use_multi_gpu:
                new_state_dict[key] = value
            elif 'module' in key and not use_multi_gpu:
                new_state_dict[key[7:]] = value
            elif 'module' not in key and use_multi_gpu:
                new_state_dict['module.'+key] = value
            else:
                new_state_dict[key] = value

        return new_state_dict

    def restore(self, use_multi_gpu):
        if self.all_args.use_single_network:
            policy_model_state_dict = self.process_state_dict(torch.load(str(self.model_dir) + '/model.pt'), use_multi_gpu)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_model_state_dict = self.process_state_dict(torch.load(str(self.model_dir) + '/actor.pt'), use_multi_gpu)
            print(type(policy_model_state_dict))
            self.policy.actor.load_state_dict(policy_model_state_dict)

    def convert_obs(self, original_obs):
        for key, value in original_obs.items():
            original_obs[key] = [value]
        obs = defaultdict(list)
        for key in original_obs.keys():
            for i in range(self.n_rollout_threads):
                obs[key].append(np.array(original_obs[key]))
        return obs

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(
                low=low,
                high=high,
                shape=shape,
                dtype=np.float32)

    def load_observation_space(self):
        observation_space = OrderedDict()
        observation_space['task_obs'] = self.build_obs_space(
                shape=(TASK_OBS_DIM,), low=-np.inf, high=np.inf)
        observation_space['rgb'] = self.build_obs_space(
                shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                low=0.0, high=1.0)
        observation_space['depth'] = self.build_obs_space(
                shape=(IMG_HEIGHT, IMG_WIDTH, 1),
                low=0.0, high=1.0)
        self.observation_space = [gym.spaces.Dict(observation_space)]
        
    def load_action_space(self):
        self.action_space = [gym.spaces.Discrete(4)]
        # if self.all_args.use_discrete_action:
        #     self.original_action_space = gym.spaces.Box(shape=(2,),
        #                                                 low=-1.0,
        #                                                 high=1.0,
        #                                                 dtype=np.float32)
        #     self.action_space = [gym.spaces.MultiDiscrete([self.all_args.num_actions, self.all_args.num_actions])]
        # else:
        #     self.action_space = [gym.spaces.Box(shape=(2,),
        #                                     low=-1.0,
        #                                     high=1.0,
        #                                     dtype=np.float32)]

    def recover_original_action(self, action):
        original_action_space = self.original_action_space
        low = original_action_space.low
        high = original_action_space.high
        splited_action_space = np.linspace(low, high, self.all_args.num_actions)
        return [splited_action_space[action[0], 0], splited_action_space[action[1], 1]]

if __name__ == "__main__":
    # obs = {
    #     'depth': np.ones((IMG_HEIGHT, IMG_WIDTH, 1)),
    #     'rgb': np.ones((IMG_HEIGHT, IMG_WIDTH, 3)),
    #     'task_obs': np.ones((TASK_OBS_DIM,))
    # }
    # agent = rMAPPOAgent(root_dir='test')
    # action = agent.act(obs)
    # print('action', action)
    a = load_args('/home/gaof/eai/igibson-challenge/onpolicy_docker_files/save/act12-ep5-mb10-bd256-hs128-TFPqk_h4_emb128/config.yaml')
