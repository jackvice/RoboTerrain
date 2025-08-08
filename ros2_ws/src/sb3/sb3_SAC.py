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
- Includes checkpoint saving and loading with proper normalization statistics
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

"""
Rover Navigation Training and Inference Script with proper cleanup handling
"""

import torch, math
import sys
import argparse
from datetime import datetime
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
#from environments.rover_env import RoverEnv
#from environments.rover_env_vision import RoverEnvVis
from environments.leo_rover_env_fused import RoverEnvFused
import rclpy


def parse_args():
    parser = argparse.ArgumentParser(description='Train or run SAC agent for rover navigation')
    parser.add_argument('--load', type=str, choices=['True', 'False'], required=True,
                      help='Whether to load from checkpoint (required for predict)')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                      help='Mode: train or predict')
    parser.add_argument('--world', type=str, choices=['inspect', 'maze', 'island', 'rubicon'], 
                      default='inspect', help='Which world to use')
    parser.add_argument('--vision', type=str, choices=['True', 'False'], default='False',
                      help='Whether to use image input')
    parser.add_argument('--checkpoint_name', type=str, help='Path to checkpoint file to load')
    parser.add_argument('--normalize_stats', type=str, 
                      help='Path to normalization statistics file (required if loading checkpoint)')
    return parser.parse_args()


def make_env(do_vision, world_name):
    def _init():
        if do_vision:
            print('Using fused image')
            #env = RoverEnvVis(world_n=world_name)
            env = RoverEnvFused(world_n=world_name)
        else:
            print('Using standard lidar model')
            env = RoverEnv(world_n=world_name)
        env = Monitor(env)
        return env
    return _init


class SaveVecNormalizeCallback(CheckpointCallback):
    """Callback for saving both model and normalization statistics"""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, env: VecNormalize):
        super().__init__(save_freq=save_freq, save_path=save_path, 
                        name_prefix=name_prefix, save_replay_buffer=False)
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save normalization statistics with matching name
            stats_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.num_timesteps}_steps_normalize.pkl"
            )
            self.env.save(stats_path)
        return super()._on_step()


def main():
    env = None
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_dir = "./checkpoints"
        tensorboard_dir = f"./tboard_logs/SAC_{args.world}_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create base environment
        env = DummyVecEnv([make_env(args.vision == 'True', args.world)])

        # Handle environment normalization
        if args.load == 'True':
            if not args.checkpoint_name:
                raise ValueError("Checkpoint name must be provided when load is True")
            if not args.normalize_stats:
                raise ValueError("Normalize stats path must be provided when loading checkpoint")
            
            # Load existing normalization stats
            env = VecNormalize.load(args.normalize_stats, env)

            # Disable updates during prediction
            if args.mode == 'predict':
                env.training = False
                env.norm_reward = False
        else:
            # Create new normalized environment
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=20.,
                clip_reward=100.,
                gamma=0.99,
                epsilon=1e-8
            )

        # Create or load model
        try:
            """
            if args.load == 'True':
                print(f"Loading model from {args.checkpoint_name}")
                model = SAC.load(args.checkpoint_name, env=env, 
                               tensorboard_log=tensorboard_dir)
            """

            if args.load == 'True':
                import torch, math
                print("Loading model …")
                model = SAC.load(args.checkpoint_name, env=env, tensorboard_log=tensorboard_dir)

                # ---- keep a copy of the previous alpha BEFORE overwriting ----------
                old_alpha = float(torch.exp(model.log_ent_coef).cpu().item())

                # ---- reset alpha ----------------------------------------------------

                new_ent_coef = 0.05
                model.log_ent_coef = torch.nn.Parameter(
                    torch.tensor(math.log(new_ent_coef), device=model.device)
                )
                model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=3e-4)
                model.target_entropy = -1.0   # optional

                print(f"α reset: {old_alpha:.5f}  →  {new_ent_coef:.5f}")
                print(f"Target entropy adjusted: -2.0 → {model.target_entropy}")

            else:
                print("Creating new model")
                model = SAC("MultiInputPolicy",
                            env,
                            device="cuda", 
                            learning_rate=3e-4,
                            buffer_size=300_000,
                            learning_starts=50_000,
                            batch_size=512,
                            train_freq=(512),  # wait until we have 2 048 new transitions
                            gradient_steps=6,         # run ~one full pass through the buffer
                            tensorboard_log=tensorboard_dir,
                            ent_coef="auto_0.5",
                            verbose=1,
                            )
                """
                model = SAC("MultiInputPolicy",
                            env,
                            device="cuda", 
                            learning_rate=3e-4,
                            tensorboard_log=tensorboard_dir,
                            buffer_size=1_000_000,
                            learning_starts=50000,
                            ent_coef="auto_0.5",
                            verbose=1,
                            batch_size=512)
                """
        except Exception as e:
            print(f"Error creating/loading model: {e}")
            raise

        # Run prediction or training
        if args.mode == 'predict':
            print("Starting prediction mode")
            obs = env.reset()
            episode_rewards = 0
            num_episodes = 0
            
            try:
                for _ in range(1_000_000):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    episode_rewards += rewards[0]
                    
                    if done:
                        print(f"Episode {num_episodes} finished with reward {episode_rewards}")
                        obs = env.reset()
                        episode_rewards = 0
                        num_episodes += 1
            except Exception as e:
                print(f"Error during prediction: {e}")
                raise
                
        else:  # Training mode
            print("Starting training mode")
            # Setup checkpoint callback
            checkpoint_callback = SaveVecNormalizeCallback(
                save_freq=100_000,
                save_path=checkpoint_dir,
                name_prefix=f"sac_{args.world}_{timestamp}",
                env=env
            )
            
            # Save initial stats
            env.save(f"{checkpoint_dir}/vec_normalize_{timestamp}_initial.pkl")
            
            try:
                model.learn(
                    total_timesteps=8_000_000,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False if args.load == 'True' else True
                )
            except Exception as e:
                print(f"Error during training: {e}")
                raise
            
    except Exception as e:
        print(f"Error in main function: {e}")
        raise
    finally:
        # Cleanup environment
        if env is not None:
            try:
                env.close()
                print("Environment closed successfully")
            except Exception as e:
                print(f"Error closing environment: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Force ROS shutdown if it's still running
        if rclpy.ok():
            try:
                rclpy.shutdown()
                print("ROS2 node shutdown successfully")
            except Exception as e:
                print(f"Error during ROS shutdown: {e}")
