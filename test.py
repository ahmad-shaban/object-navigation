"""
test the saved model on the environment
"""
import gym
from gym_minigrid.wrappers import ImgObsWrapper
from mini_behavior.utils.wrappers import MiniBHFullyObsWrapper
from mini_behavior.register import register
import mini_behavior

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import json

env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
