import sys
import argparse
from datetime import datetime
import os
ros_path = '/opt/ros/humble/lib/python3.10/site-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.rover_environment_pointnav import RoverEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_features_extractor import CustomCombinedExtractor

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent for rover navigation')
    parser.add_argument('--load', type=str, choices=['True', 'False'], required=True,
                      help='Whether to load from checkpoint')
    parser.add_argument('--checkpoint_name', type=str,
                      help='Path to checkpoint file to load')
    return parser.parse_args()

def main():
    args = parse_args()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(),
    )
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Set up environment
    env = RoverEnv()
    
    # Set up directories
    checkpoint_dir = "./checkpoints"
    tensorboard_dir = f"./tboard_logs/SAC_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.load == 'True':
        if not args.checkpoint_name:
            raise ValueError("Checkpoint name must be provided when load is True")
        # Load existing model
        model = SAC.load(args.checkpoint_name, 
                        env=env, 
                        tensorboard_log=tensorboard_dir,
                        policy_kwargs=policy_kwargs)
    else:
        # Create new model
        model = SAC("MultiInputPolicy",
                    env,
                    tensorboard_log=tensorboard_dir,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,          # SAC often needs a different learning rate
                    buffer_size=1_000_000,       # Replay buffer size
                    learning_starts=20000,       # Collect this many steps before training
                    batch_size=256,              # Typical SAC batch size
                    tau=0.005,                   # Soft update coefficient
                    gamma=0.99,                  # Discount factor
                    train_freq=1,                # Update policy every N steps
                    gradient_steps=1,            # How many gradient steps per update
                    ent_coef='auto'             # Automatic entropy tuning
                    )
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix=f"sac_rover_model_{timestamp}",
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
