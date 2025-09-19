# l2_tactic/feature_extractors.py
"""
Custom feature extractors for FinRL models
"""
import torch
import torch.nn as nn
from loguru import logger

# Import for Claude model custom feature extractor
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym
except ImportError:
    BaseFeaturesExtractor = None
    gym = None
    logger.warning("stable_baselines3/gymnasium not available")


if BaseFeaturesExtractor is not None:
    class RiskAwareExtractor(BaseFeaturesExtractor):
        """
        Custom feature extractor inspired by the paper's risk-aware architecture
        Used for Claude model training and inference
        """

        def __init__(self, observation_space, features_dim=512):
            if gym is None:
                raise ImportError("gymnasium not available")
            if hasattr(gym, 'spaces') and hasattr(gym.spaces, 'Box'):
                if not isinstance(observation_space, gym.spaces.Box):
                    raise ValueError("observation_space must be a gym.spaces.Box")
            super(RiskAwareExtractor, self).__init__(observation_space, features_dim)

            n_input_features = observation_space.shape[0]

            # Multi-layer feature extraction network
            self.feature_net = nn.Sequential(
                nn.Linear(n_input_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
                nn.ReLU()
            )

        def forward(self, observations):
            return self.feature_net(observations)
else:
    # Fallback class when BaseFeaturesExtractor is not available
    class RiskAwareExtractor:
        def __init__(self, observation_space, features_dim=512):
            raise ImportError("BaseFeaturesExtractor not available - gymnasium/stable_baselines3 not installed")
