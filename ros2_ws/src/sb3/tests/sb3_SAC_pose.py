import sys
ros_path = '/opt/ros/humble/lib/python3.10/site-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

from stable_baselines3 import SAC
from environments.rover_environment import RoverEnv
from environments.pose-converter import PoseConverterNode 
from stable_baselines3.common.callbacks import CheckpointCallback

def run_ros_node(node):
    """Function to run the ROS node in a separate thread"""
    rclpy.spin(node)

def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create the pose converter node
    pose_node = PoseConverterNode()
    
    # Create and start ROS spinning thread
    executor = MultiThreadedExecutor()
    executor.add_node(pose_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # Create the environment
    env = RoverEnv()

    # Create and train the SAC agent with some tuned hyperparameters
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=2048,
        tau=0.005,
        gamma=0.99,
        gradient_steps=1,        # Automatically adjust gradient steps
        learning_starts=20_000,        # Collect this many steps before training starts
        ent_coef="auto",             # Automatically adjust entropy coefficient
        verbose=1,
        tensorboard_log="./tboard_logs/"  # For monitoring training
    )

    # Create a callback that saves the model every 50000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="sac_rover_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Train the model
    model.learn(
        total_timesteps=2_000_000,
        callback=checkpoint_callback
    )

    # Save the final model
    model.save("sac_rover_final")

    #cleanup
    node.destroy_node()
    rclpy.shutdown()
    ros_thread.join()
    return 1


if __name__ == "__main__":
    main()

