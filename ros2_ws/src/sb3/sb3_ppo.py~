

import sys
ros_path = '/opt/ros/humble/lib/python3.10/site-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.rover_environment_pointnav import RoverEnv
#from environments.pose-converter import PoseConverterNode 
from stable_baselines3.common.callbacks import CheckpointCallback
# Create and wrap the environment

env = RoverEnv()


# Create and train the PPO agent
model = PPO("MultiInputPolicy",
            env,
            tensorboard_log="./tboard_logs/",  #
            verbose=1
            )
checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="ppo_rover_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

model.learn(
    total_timesteps=2_000_000,
    callback=checkpoint_callback
)
