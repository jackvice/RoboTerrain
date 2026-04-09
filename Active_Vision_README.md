# Active Vision for Social Navigation (AVSN)

A joint reinforcement learning framework that unifies locomotion and discrete gaze control for socially aware robot navigation in unstructured environments. AVSN learns *where to look* and *where to drive* within a single end-to-end policy, using a latent world model to solve the credit-assignment problem inherent in active sensing.

**Paper:** [Active Vision for Social Navigation](https://www.preprints.org/manuscript/202604.0068)

---

## Overview

The system spans three repositories under `~/src/`:

| Repository | Role |
|---|---|
| **RoboTerrain** | Gazebo Fortress simulation, Leo Rover with fisheye camera, dynamic human actors, metrics logging |
| **attention** | Spatiotemporal attention model — predicts near-future human occupancy heatmaps from RGB + YOLO masks |
| **dreamerJMV3** | DreamerV3-based RL agent — consumes fused observations (grayscale, depth, attention heatmap) and outputs joint locomotion + gaze actions |

### Pipeline

```
Fisheye camera (6 rectified windows, ~160° FOV)
        │
        ▼
fisheye_ros2_mem_share.py ──► shared memory
        │
        ▼
inference.py
  ├─ YOLO pedestrian detection
  ├─ Spatiotemporal attention heatmap
  ├─ Monocular depth estimation
  └─ Fused 3-channel observation ──► shared memory
        │
        ▼
DreamerV3 (dreamerv3/main.py)
  ├─ World model learns gaze–observation dynamics
  ├─ Joint action: [linear_vel, angular_vel, pan_window]
  └─ Pan selects from 5 yaw offsets: −60°, −30°, 0°, +30°, +60°
```

---

## Prerequisites

- **Ubuntu 22.04** with [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) and [Gazebo Fortress](https://gazebosim.org/docs/latest/ros_installation/)
- **NVIDIA GPU** with CUDA (for YOLO, depth estimation, and JAX/DreamerV3)
- Python 3.10+

### Python Environments

The attention pipeline and DreamerV3 have conflicting dependencies (notably different JAX versions), so they **must** run in separate Python environments. Use whichever environment manager you prefer (venv, conda, virtualenv, etc.). Each repository contains a `requirements.txt`.

```bash
# Attention environment
cd ~/src/attention
# create and activate a Python 3.10+ environment, then:
pip install -r requirements.txt

# DreamerV3 environment
cd ~/src/dreamerJMV3
# create and activate a separate Python 3.10+ environment, then:
pip install -r requirements.txt
```

### Build the ROS 2 Workspace

```bash
source /opt/ros/humble/setup.bash
cd ~/src/RoboTerrain/ros2_ws
colcon build
source install/setup.bash
```

---

## Directory Layout

```
~/src/
├── RoboTerrain/
│   └── ros2_ws/src/
│       ├── roverrobotics_ros2/       # Leo Rover model, fisheye camera, Gazebo worlds
│       ├── dynamic_obstacles/        # Human actor spawner + trajectory files
│       └── rover_metrics/            # Navigation metrics logger
├── attention/
│   ├── inference/
│   │   └── fisheye_ros2_mem_share.py # ROS2 → shared memory bridge
│   ├── inference.py                  # YOLO + attention + depth fusion
│   ├── trajectory_model.py           # Spatiotemporal attention model (Flax/JAX)
│   ├── preprocess_dataset.py         # Dataset preprocessing
│   ├── run_efficient_training.py     # Training entry point
│   └── model_output/                 # Trained checkpoints
└── dreamerJMV3/
    ├── dreamerv3/
    │   ├── main.py                   # DreamerV3 entry point
    │   └── configs.yaml              # Contains `leorover` config
    └── logdir/                       # Training logs and checkpoints
```

---

## Running the Full System

The system requires **five terminals**. Source ROS 2 in every terminal that uses ROS:

```bash
source /opt/ros/humble/setup.bash
source ~/src/RoboTerrain/ros2_ws/install/setup.bash
```

### Terminal 1 — Gazebo Simulation

Launch the Leo Rover with fisheye camera in one of the available worlds (inspect, construction, island):

```bash
# With GUI
ros2 launch roverrobotics_gazebo Leo_rover_fisheye.launch.py

# Headless (for training runs)
ros2 launch roverrobotics_gazebo Leo_rover_fisheye.launch.py headless:=true
```

### Terminal 2 — Fisheye Camera Bridge

Subscribes to the fisheye camera topic, rectifies six 320×320 windows, and writes them to shared memory:

```bash
cd ~/src/attention/inference
python fisheye_ros2_mem_share.py
```

### Terminal 3 — Attention + Perception Pipeline

Runs YOLO detection, spatiotemporal attention, and monocular depth estimation. Outputs the fused observation to shared memory for DreamerV3:

```bash
# activate your attention Python environment, then:
cd ~/src/attention
python inference.py --attention_mode ./model_output/checkpoint_epoch_1000.pkl
```

### Terminal 4 — DreamerV3 RL Agent

Launches the DreamerV3 agent with the `leorover` config. The `LD_LIBRARY_PATH` filtering prevents environment-installed libraries from conflicting with ROS 2 system libraries:

```bash
# activate your DreamerV3 Python environment, then:

# Filter LD_LIBRARY_PATH to ROS-only paths
FILTERED_LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E '^/opt/ros' | tr '\n' ':' | sed 's/:$//')

env LD_LIBRARY_PATH="$FILTERED_LD_LIBRARY_PATH" \
    CUDA_HOME="" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    python dreamerv3/main.py \
      --configs leorover \
      --logdir ./logdir/dreamer/{timestamp}
```

### Terminal 5 — Dynamic Human Actors

Spawn pedestrian actors with predefined trajectories (examples shown for the inspect world):

```bash
cd ~/src/RoboTerrain/ros2_ws/src/dynamic_obstacles

python spawn.py \
  --trajectory_file trajectories/inspect_linear.sdf \
  --world_name inspect \
  --actor_name linear

python spawn.py \
  --trajectory_file trajectories/inspect_diag.sdf \
  --world_name inspect \
  --actor_name diag

python spawn.py \
  --trajectory_file trajectories/inspect_corner_triangle.sdf \
  --world_name inspect \
  --actor_name triangle
```

---

## Training the Attention Model

The spatiotemporal attention model is trained separately on datasets of RGB frames with YOLO-generated pedestrian masks. Training produces checkpoint files consumed by `inference.py`.

```bash
# activate your attention Python environment, then:
cd ~/src/attention

python run_efficient_training.py \
  --dataset_path /path/to/data \
  --output_dir ./model_output \
  --preprocessed_dir ./preprocessed_data \
  --num_epochs 1000 \
  --batch_size 8 \
  --sequence_length 5 \
  --learning_rate 1e-4 \
  --target_width 320 \
  --target_height 320 \
  --yolo_model_path /path/to/yolo11n.onnx \
  --embedding_dim 128 \
  --num_heads 4 \
  --tensorboard_dir ./log_dir
```

Resume from a checkpoint:

```bash
python run_efficient_training.py \
  --dataset_path /path/to/data \
  --resume_checkpoint ./model_output/checkpoint_epoch_500.pkl \
  # ... other args as above
```

Monitor training:

```bash
tensorboard --logdir ./log_dir
```

---

## Metrics and Evaluation

Navigation metrics are logged via the ROS 2 metrics node:

```bash
ros2 run rover_metrics metrics_node
```

Visualize and compare runs:

```bash
cd ~/src/RoboTerrain/metrics_analyzer
python cli_main.py \
  data/metric_logs/run1.csv \
  data/metric_logs/run2.csv \
  -m CV IM -p time_series
```

Available metrics: Total Collisions (TC), Obstacle Clearance (OC), Current Velocity (CV), IMU Acceleration (IM), Rough Terrain (RT), Distance Traveled (DT), Smoothness (SM).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No camera frames in shared memory | Confirm Gazebo is running and `fisheye_ros2_mem_share.py` is receiving on the camera topic |
| YOLO ONNX errors | Ensure `onnxruntime-gpu` is installed in the `attent` environment |
| DreamerV3 CUDA OOM | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (already in the launch command above) |
| `libstdc++` version conflicts | The `LD_LIBRARY_PATH` filter in Terminal 4 resolves most Python-environment/ROS library clashes |
| Actors not visible | Verify the `--world_name` flag matches the launched Gazebo world |
| Checkpoint load failure | Checkpoints are Flax `.pkl` files; ensure the config (embedding_dim, num_heads) matches |

---

## Citation

```bibtex
@article{vice2025avsn,
  title   = {Active Vision for Social Navigation},
  author  = {Vice, Jack M. and Sukthankar, Gita},
  journal = {Preprints},
  year    = {2025},
  doi     = {10.20944/preprints202604.0068}
}

@article{vice2025dune,
  title     = {DUnE: A Versatile Dynamic Unstructured Environment for Off-Road Navigation},
  author    = {Vice, Jack M. and Sukthankar, Gita},
  journal   = {Robotics},
  volume    = {14},
  number    = {4},
  pages     = {35},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/robotics14040035}
}
```

## License

RoboTerrain is Apache 2.0. See individual repositories for third-party license details (ONNX Runtime, YOLO, Leo Rover MIT, Clearpath BSD).
