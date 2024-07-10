#!.venv/bin/python
import multiprocessing

import gym
from gym_minigrid.wrappers import ImgObsWrapper
# from mini_behavior.utils.wrappers import MiniBHFullyObsWrapper
from mini_behavior.register import register
import mini_behavior
from sb3_contrib import RecurrentPPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import json
from object_nav import envs

from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn.functional as F

import os

script_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=False, help='name of task to train on')
parser.add_argument("--partial_obs", default=True)
parser.add_argument("--room_size", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--total_timesteps", type=int, default=1e6)
parser.add_argument("--dense_reward", action="store_true")
parser.add_argument("--policy_type",
                    # default="CnnPolicy",
                    default="MlpLstmPolicy"
                    )
parser.add_argument(
    "--auto_env",
    default=True,
    help='flag to procedurally generate floorplan'
)
# NEW
parser.add_argument(
    "--auto_env_config",
    help='Path to auto environment JSON file',
    default='./object_nav/envs/floorplans/one_room.json'
)
args = parser.parse_args()
partial_obs = args.partial_obs


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        # Create the CNN feature extractor
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]
        n_flatten += env.observation_space["mission"].shape[0] + 1  # Add one for direction

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, observations):
        # Extract features from the image
        image, direction, mission = observations["image"], observations["direction"], observations["mission"]
        image_features = self.cnn(image)
        # convert direction value into one-hot vector
        direction = F.one_hot(direction, num_classes=4)
        direction = direction.unsqueeze(0) if len(list(direction.shape)) < 2 else direction
        # print(image_features.shape, mission.shape, direction.shape)

        # Concatenate features with one-hot encoded mission and direction
        concatenated_features = torch.cat((image_features, mission, direction), dim=1)

        return self.linear(concatenated_features)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=128)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                # Compute shape by doing one forward pass
                with torch.no_grad():
                    n_flatten = extractors[key](torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]
                total_concat_size += n_flatten
            elif key == "mission":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 32)
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size + 4  # 4 for direction

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            out = extractor(observations[key])
            out = out.unsqueeze(0) if len(list(out.shape)) < 2 else out
            encoded_tensor_list.append(out)
        direction = F.one_hot(observations["direction"].long(), num_classes=4).float()
        if len(direction.shape) == 3:
            direction = direction.squeeze(1)
        encoded_tensor_list.append(direction)        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


# policy_kwargs = dict(
#     features_extractor_class=MinigridFeaturesExtractor,
#     features_extractor_kwargs=dict(features_dim=128),
# )
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)

# Env wrapping
env_name = "MiniGrid-igridson-16x16-N2-v0"

# wandb init
config = {
    "policy_type": args.policy_type,
    "total_timesteps": args.total_timesteps,
    "env_name": env_name,
}

print('init wandb')
run = wandb.init(
    entity="rl_mo",
    project=env_name,
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Add this line
    # Your code for creating vectorized environment and training here

    print('make env')

    env = gym.make(env_name)

    print('begin training')
    def clip_range_schedule(progress):
        """
        This function defines a linear decay schedule for the clip range.
        Args:
            progress (float): The remaining training progress (0.0 to 1.0).
        Returns:
            float: The adjusted clip range value based on progress.
        """
        start_clip_range = 0.3  # Initial clip range value
        min_clip_range = 0.05  # Minimum clip range value

        return start_clip_range - (start_clip_range - min_clip_range) * progress

    # Policy training
    model = RecurrentPPO(config["policy_type"], env, clip_range=clip_range_schedule, gamma=0.99, n_steps=128,
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"./runs/{run.id}")


    def custom_lr(progress):
        clip_range = clip_range_schedule(progress)
        # Adjust learning rate based on clip range (example)
        return 0.005 * (1.0 - clip_range)  # Adjust based on your logic


    model.learning_rate = custom_lr  # Set the custom learning rate scheduler

    model.learn(config["total_timesteps"], callback=WandbCallback(model_save_path=f"models/{run.id}"))

    dense_reward = '-'
    if not partial_obs:
        model.save(f"models/ppo_cnn/{env_name}{dense_reward}")
    else:
        model.save(f"models/ppo_recurrent_dense/{env_name}{dense_reward}")

    run.finish()
