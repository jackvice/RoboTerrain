import sys
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
import numpy as np

ros_path = '/opt/ros/humble/lib/python3.10/site-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

from stable_baselines3 import PPO
from environments.rover_environment import RoverEnv
from environments.pose_converter import PoseConverterNode
from stable_baselines3.common.callbacks import CheckpointCallback

def main(args=None):
    # Initialize ROS once
    rclpy.init(args=args)
    
    # Create the pose converter node
    pose_node = PoseConverterNode()
    
    # Create and start ROS spinning thread
    executor = MultiThreadedExecutor()
    executor.add_node(pose_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    try:
        # Create the environment and pass the nodes
        env = RoverEnv(init_ros=False)  # Tell RoverEnv not to initialize ROS
        env.pose_node = pose_node
        
        # Create and train the PPO agent
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            batch_size=2048,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./tboard_logs/"
        )
        
        # Create a callback that saves the model
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path="./checkpoints/",
            name_prefix="ppo_rover_model",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        
        # Train the model
        model.learn(
            total_timesteps=2_000_000,
            callback=checkpoint_callback
        )
        
        # Save the final model
        model.save("ppo_rover_final")
        
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Cleanup
        pose_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == "__main__":
    main()
