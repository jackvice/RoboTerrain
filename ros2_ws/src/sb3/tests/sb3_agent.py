

import sys
ros_path = '/opt/ros/humble/lib/python3.10/site-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.rover_environment import RoverEnv 

# Create and wrap the environment
env = RoverEnv()
env = DummyVecEnv([lambda: env])

# Create and train the PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
