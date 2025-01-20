import sys
import argparse
from datetime import datetime
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn

from environments.rover_env import RoverEnv
#from environments.inspection_env import RoverEnv
#from environments.island_env import RoverEnv

from stable_baselines3.common.callbacks import CheckpointCallback
#from custom_features_extractor import CustomCombinedExtractor
from stable_baselines3.common.monitor import Monitor
#from logging_wrapper import LoggingWrapper  # Add this with other imports


def parse_args():
    parser = argparse.ArgumentParser(description='Train or run PPO agent for rover navigation')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                      help='Mode: train or predict')
    parser.add_argument('--load', type=str, choices=['True', 'False'], default='True',
                      help='Whether to load from checkpoint (required for predict)')
    parser.add_argument('--checkpoint_name', type=str, help='Path to checkpoint file to load')
    return parser.parse_args()


def make_env(world_name):
    def _init():
        env = RoverEnv(world_n=world_name)
        env = Monitor(env)
        return env
    return _init


def mainNew():
    args = parse_args()
    #world_name = 'inspect'
    #world_name = 'maze'
    world_name = 'moon'
    
    # Set up environment
    env = DummyVecEnv([make_env(world_name)])  # Pass a list with make_env function
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
        model = SAC.load(args.checkpoint_name, env=env)
    else:
        raise ValueError("Inference requires a checkpoint to load")

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
    else:  # Training Mode
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Set up directories
        checkpoint_dir = "./checkpoints"
        tensorboard_dir = f"./tboard_logs/SAC_{world_name}_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create new model if not loading
        if args.load == 'False':
            model = SAC("MultiInputPolicy",
                        env,
                        learning_rate=get_linear_fn(3e-4, 5e-5, 1.0),
                        tensorboard_log=tensorboard_dir,
                        buffer_size=1_000_000,
                        learning_starts=50000,
                        ent_coef="auto_0.5",
                        verbose=1,
                        batch_size=512)

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
            total_timesteps=10_000_000,
            callback=checkpoint_callback,
            reset_num_timesteps=False if args.load == 'True' else True
        )


def main():
    args = parse_args()
    #world_name = 'inspect'
    world_name = 'moon'
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Set up environment
    env = DummyVecEnv([make_env(world_name)])  # Note: Pass a list with make_env function
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
    tensorboard_dir = f"./tboard_logs/SAC_{world_name}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
        total_timesteps=10_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False if args.load == 'True' else True
    )

if __name__ == "__main__":
    main()


