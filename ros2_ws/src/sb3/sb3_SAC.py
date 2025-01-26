"""
Rover Navigation Training and Inference Script
-------------------------------------------
This script implements training and inference for a Soft Actor-Critic (SAC) 
reinforcement learning agent designed for rover navigation tasks. The agent can be 
trained in different environments (inspect, maze, island, rubicon) with optional 
vision-based input.

Features:
- Supports both standard state-based and vision-based inputs
- Multiple world environments: inspect, maze, island, rubicon
- Implements SAC (Soft Actor-Critic) with auto-tuned entropy
- Includes checkpoint saving and loading
- Uses vectorized environments with observation/reward normalization
- Tensorboard logging support

Usage:
    python script.py --mode [train/predict] 
                    --load [True/False] 
                    --world [inspect/maze/island/rubicon]
                    --vision [True/False]
                    --checkpoint_name [path_to_checkpoint]

Requirements:
    - stable-baselines3
    - gym
    - numpy
    - tensorboard

Author: Jack Vice
Last Updated: 01/26/25
"""

import sys
import argparse
from datetime import datetime
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from environments.rover_env import RoverEnv
from environments.rover_env_vision import RoverEnvVis
from stable_baselines3.common.callbacks import CheckpointCallback
#from custom_features_extractor import CustomCombinedExtractor
from stable_baselines3.common.monitor import Monitor

print('1')

def parse_args():
    parser = argparse.ArgumentParser(description='Train or run PPO agent for rover navigation')
    parser.add_argument('--load', type=str, choices=['True', 'False'], required=True,
                      help='Whether to load from checkpoint (required for predict)')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                      help='Mode: train or predict')
    parser.add_argument('--world', type=str, choices=['inspect', 'maze', 'island', 'rubicon'], default='inspect',
                      help='Which world to use (inspect, maze, island, rubicon)')
    parser.add_argument('--vision', type=str, choices=['True', 'False'], default='False',
                      help='Whether to use image input')
    parser.add_argument('--checkpoint_name', type=str, help='Path to checkpoint file to load')
    return parser.parse_args()


def make_env(do_vision, world_name):
    def _init():
        if do_vision:
            env = RoverEnvVis(world_n=world_name)
        else:
            env = RoverEnv(world_n=world_name)
        env = Monitor(env)
        return env
    return _init


def main():
    rclpy.init()
    args = parse_args()
    world_name = args.world
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_dir = "./checkpoints"
    tensorboard_dir = f"./tboard_logs/SAC_{world_name}_{timestamp}"
    print('1.1')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print('2')

    learning_rate = 3e-4 if args.vision else get_linear_fn(3e-4, 2e-4, 1.0)

    
    # Set up environment
    env = DummyVecEnv([make_env(args.vision, world_name)])  # Note: Pass a list with make_env function
    env = VecNormalize(
        env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards
        clip_obs=20.,
        clip_reward=100.,
        gamma=0.99,
        epsilon=1e-8
    )
    
    if args.load == 'True':
        if not args.checkpoint_name:
            raise ValueError("Checkpoint name must be provided when load is True")
        # Load existing model
        model = SAC.load(args.checkpoint_name, 
                        env=env, 
                        tensorboard_log=tensorboard_dir)
                        #policy_kwargs=policy_kwargs)
    else:
        # Create new model
        model = SAC("MultiInputPolicy",
                    env,
                    learning_rate = get_linear_fn(3e-4, 5e-5, 1.0),  # Starts at 3e-4, decays to 5e-5
                    tensorboard_log=tensorboard_dir,
                    buffer_size = 1_000_000,  # 1e6
                    learning_starts = 50000,
                    ent_coef = "auto_0.5",
                    verbose=1,
                    batch_size=512,
                    )       
    print('3')
    if args.mode == 'predict':
        obs = env.reset()
        done = False
        for _ in range(1_000_000):
            # Predict action from model
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()
        env.close()
    else:

        # Set up checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=200_000,
            save_path=checkpoint_dir,
            name_prefix=f"sac_{world_name}_{timestamp}",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
    
        # Train model
        model.learn(
            total_timesteps=5_000_000,
            callback=checkpoint_callback,
            reset_num_timesteps=False if args.load == 'True' else True
        )


