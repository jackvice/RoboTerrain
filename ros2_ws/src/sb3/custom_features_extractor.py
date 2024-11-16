import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for different observation types with appropriate feature sizes."""
    
    def __init__(self, observation_space: gym.spaces.Dict):
        # Initialize with dummy features_dim (will be updated later)
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Check if observation_space is of type Dict
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(f"Expected gym.spaces.Dict, but got {type(observation_space)}")
 
        
        extractors = {}
        total_concat_size = 0
        
        # Define appropriate feature sizes for each observation type
        feature_sizes = {
            "lidar": 128,  # Complex spatial data, keep at 128
            "pose": 3,     # Simple positional data (x,y,z)
            "imu": 3,      # Simple rotational data (roll,pitch,yaw)
            "target": 2    # Simple target data (distance,angle)
        }
        
        # Create neural networks for each observation type
        for key, subspace in observation_space.spaces.items():
            if key == "lidar":
                # 1D CNN for lidar data
                cnn = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                # Calculate the size of flattened features
                test_tensor = th.zeros(1, 1, subspace.shape[0])  # [batch, channels, length]
                with th.no_grad():
                    n_flatten = cnn(test_tensor).shape[1]
                
                # Add final fully connected layer
                fc = nn.Sequential(
                    nn.Linear(n_flatten, feature_sizes[key]),
                    nn.ReLU()
                )
                extractors[key] = nn.Sequential(cnn, fc)
                
            elif key in ["pose", "imu", "target"]:
                # Simple MLPs for vector inputs with smaller feature sizes
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_sizes[key] * 2),  # Wider intermediate layer
                    nn.ReLU(),
                    nn.Linear(feature_sizes[key] * 2, feature_sizes[key]),
                    nn.ReLU()
                )
            else:
                raise ValueError(f"Unknown observation key: {key}")
                
            total_concat_size += feature_sizes[key]
            
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size  # Now equals 128 + 8 + 8 + 8 = 152

    def forward(self, observations) -> th.Tensor:
        """Process each observation type through its respective extractor."""
        encoded_tensor_list = []
        
        # Process each observation type
        for key, extractor in self.extractors.items():
            if key == "lidar":
                # Add channel dimension for CNN
                x = observations[key].unsqueeze(1)  # [batch, channel, length]
            else:
                x = observations[key]
            encoded_tensor_list.append(extractor(x))
            
        # Concatenate all features
        return th.cat(encoded_tensor_list, dim=1)


    
class CustomCombinedExtractorOLD(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Initialize with dummy features_dim (will be updated later)
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        feature_size = 128  # Size of each feature output
        
        # Create neural networks for each observation type
        for key, subspace in observation_space.spaces.items():
            if key == "lidar":
                # 1D CNN for lidar data
                cnn = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                # Calculate the size of flattened features
                test_tensor = th.zeros(1, 1, subspace.shape[0])  # [batch, channels, length]
                with th.no_grad():
                    n_flatten = cnn(test_tensor).shape[1]
                
                # Add final fully connected layer
                fc = nn.Sequential(
                    nn.Linear(n_flatten, feature_size),
                    nn.ReLU()
                )
                extractors[key] = nn.Sequential(cnn, fc)
                
            elif key in ["pose", "imu", "target"]:
                # Simple MLPs for vector inputs
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size),
                    nn.ReLU(),
                    nn.Linear(feature_size, feature_size),
                    nn.ReLU()
                )
            else:
                raise ValueError(f"Unknown observation key: {key}")
                
            total_concat_size += feature_size
            
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        
        # Process each observation type
        for key, extractor in self.extractors.items():
            if key == "lidar":
                # Add channel dimension for CNN
                x = observations[key].unsqueeze(1)  # [batch, channel, length]
            else:
                x = observations[key]
            encoded_tensor_list.append(extractor(x))
            
        # Concatenate all features
        return th.cat(encoded_tensor_list, dim=1)
