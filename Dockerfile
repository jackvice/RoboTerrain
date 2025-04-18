# Base image with Ubuntu 22.04 (required for ROS 2 Humble)
FROM osrf/ros:humble-desktop

# Set up environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive


# Install necessary OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglvnd-dev \
    mesa-utils


# Install ROS 2 Humble full desktop and necessary dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-diff-drive-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-xacro \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Fortress and ROS-Gazebo integration
RUN apt-get update && apt-get install -y \
    ros-humble-ros-gz \
    ros-humble-ros-gz-sim \
    ros-humble-gazebo-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (modify for CPU-only if needed)
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install additional Python libraries for reinforcement learning and PPO
RUN pip3 install gym==0.23.1 stable-baselines3[extra] numpy scipy

# Set up ROS 2 workspace in /home/ros2_ws
WORKDIR /home/ros2_ws
RUN mkdir -p src

# Copy everything from the host's ros2_ws into the container
COPY ros2_ws/src src/

# Adjust permissions
RUN chmod -R a+rw /home/ros2_ws/src

# Build ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Add environment setup to bashrc for convenience
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /home/ros2_ws/install/setup.bash" >> ~/.bashrc
ENV PATH="/opt/ros/humble/bin:${PATH}"

# Expose Gazebo and ROS 2 ports
EXPOSE 11311 11345

# Set entrypoint to keep container alive and workspace sourced
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /home/ros2_ws/install/setup.bash && tail -f /dev/null"]
