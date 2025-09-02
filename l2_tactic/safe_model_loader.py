# l2_tactic/safe_model_loader.py
import os
import logging
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    try:
        logger.info(f"Intentando carga desde {model_path}")
        policy_path = os.path.join(model_path, "policy.pth")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"policy.pth no encontrado en {model_path}")
        
        import torch
        from stable_baselines3 import PPO
        import gymnasium as gym
        from gymnasium import spaces
        
        # Inferir dimensiones
        policy_state = torch.load(policy_path, map_location='cpu', weights_only=True)
        obs_dim, action_dim = None, None
        for key, tensor in policy_state.items():
            if 'mlp_extractor' in key and 'weight' in key and tensor.dim() == 2:
                obs_dim = tensor.shape[1]
                break
        for key, tensor in policy_state.items():
            if 'action_net' in key and 'weight' in key:
                action_dim = tensor.shape[0]
                break
        if obs_dim is None or action_dim is None:
            action_dim = 2  # Asumir 2 acciones por defecto
            obs_dim = 28    # Asumir 28 observaciones por defecto
        
        # Crear entorno dummy
        class DummyEnv(gym.Env):
            def __init__(self, obs_dim, action_dim):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(action_dim)
            def step(self, action):
                return np.zeros(self.observation_space.shape), 0, True, False, {}
            def reset(self, **kwargs):
                return np.zeros(self.observation_space.shape), {}
        
        dummy_env = DummyEnv(obs_dim, action_dim)
        
        # Crear modelo PPO
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )
        model = PPO('MlpPolicy', dummy_env, policy_kwargs=policy_kwargs, verbose=0, device='cpu')
        
        # Cargar estado
        filtered_policy_dict = {k: v for k, v in policy_state.items() if k != 'log_std'}
        model.policy.load_state_dict(filtered_policy_dict, strict=False)
        logger.info("✅ Modelo cargado desde policy.pth")
        return model
    
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        raise