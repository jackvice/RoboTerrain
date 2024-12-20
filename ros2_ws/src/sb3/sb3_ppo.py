import sys
import argparse
from datetime import datetime
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environments.rover_env_heading_vel import RoverEnv
from stable_baselines3.common.callbacks import CheckpointCallback
#from custom_features_extractor import CustomCombinedExtractor
from stable_baselines3.common.monitor import Monitor
#from logging_wrapper import LoggingWrapper  # Add this with other imports


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent for rover navigation')
    parser.add_argument('--load', type=str, choices=['True', 'False'], required=True,
                      help='Whether to load from checkpoint')
    parser.add_argument('--checkpoint_name', type=str,
                      help='Path to checkpoint file to load')
    return parser.parse_args()

def make_env():
    def _init():
        env = RoverEnv()
        env = Monitor(env)
        return env
    return _init

def main():
    args = parse_args()

    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Set up environment
    env = DummyVecEnv([make_env()])  # Note: Pass a list with make_env function
    env = VecNormalize(
        env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards
        clip_obs=20.,
        clip_reward=100.,
        gamma=0.99,
        epsilon=1e-8
    )
    
    # Set up directories
    checkpoint_dir = "./checkpoints"
    tensorboard_dir = f"./tboard_logs/PPO_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.load == 'True':
        if not args.checkpoint_name:
            raise ValueError("Checkpoint name must be provided when load is True")
        # Load existing model
        model = PPO.load(args.checkpoint_name, 
                        env=env, 
                        tensorboard_log=tensorboard_dir)
                        #policy_kwargs=policy_kwargs)
    else:
        # Create new model
        model = PPO("MultiInputPolicy",
                    env,
                    tensorboard_log=tensorboard_dir,
                    verbose=1,
                    n_steps=4096,         # increase for GPU
                    batch_size=256)       # increase for GPU

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_zero_{timestamp}",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # Train model
    model.learn(
        total_timesteps=2_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False if args.load == 'True' else True
    )

if __name__ == "__main__":
    main()
