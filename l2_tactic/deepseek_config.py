import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# DeepSeek model configuration
config = {
    'policy_kwargs': {
        'activation_fn': torch.nn.ReLU,
        'net_arch': [512, 256, 128]
    }
}

class DeepSeekFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1250):
        super().__init__(observation_space, features_dim)
        import torch.nn as nn
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1250),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)

class DeepSeekPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Apply config
        if 'net_arch' not in kwargs:
            kwargs['net_arch'] = config['policy_kwargs']['net_arch']
        if 'activation_fn' not in kwargs:
            kwargs['activation_fn'] = config['policy_kwargs']['activation_fn']
        if 'share_features_extractor' not in kwargs:
            kwargs['share_features_extractor'] = False
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = DeepSeekFeaturesExtractor
        super().__init__(*args, **kwargs)
