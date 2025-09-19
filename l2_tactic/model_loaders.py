# l2_tactic/model_loaders.py
"""
Model loading utilities for different FinRL model formats
"""
import os
from loguru import logger

# Import for model loading
try:
    from stable_baselines3 import PPO
    import gymnasium as gym
except ImportError:
    PPO = None
    gym = None
    logger.warning("stable_baselines3/gymnasium not available")


class ModelLoaders:
    """Collection of methods for loading different model formats"""

    @staticmethod
    def check_model_file(model_path: str) -> bool:
        """Check if model file exists and is valid"""
        if not os.path.exists(model_path):
            logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
            return False
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Menos de 1KB
            logger.error(f"❌ Archivo de modelo demasiado pequeño ({file_size} bytes): {model_path}")
            return False
        logger.info(f"✅ Archivo de modelo válido ({file_size/1024:.1f}KB): {model_path}")
        return True

    @staticmethod
    def load_stable_baselines3_model(zip_path: str) -> bool:
        """Load stable_baselines3 PPO model from ZIP with correct architecture and device"""
        try:
            if PPO is None:
                logger.error("stable_baselines3 not available")
                return False

            import torch

            # Check if this is the grok.zip model which has different stored parameters
            if "grok.zip" in zip_path:
                # Load grok.zip without policy_kwargs to avoid mismatch with stored parameters
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading grok.zip model from: {zip_path} with device={device} (no policy_kwargs to avoid mismatch)")
                model = PPO.load(zip_path, device=device)
                logger.info(f"Grok model loaded successfully! Policy: {type(model.policy)}")
                return model
            else:
                # For other models, use the original policy_kwargs
                policy_kwargs = dict(
                    activation_fn=torch.nn.ReLU,
                    net_arch=dict(pi=[256, 256], vf=[256, 256])
                )
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading stable_baselines3 PPO model from: {zip_path} with device={device} and policy_kwargs={policy_kwargs}")
                model = PPO.load(zip_path, device=device, policy_kwargs=policy_kwargs)
                logger.info(f"PPO model loaded successfully via stable_baselines3! Policy: {type(model.policy)}")
                return model
        except Exception as e:
            logger.error(f"Error loading stable_baselines3 model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_pickle_model(pkl_path: str):
        """Load pickled model"""
        try:
            import pickle
            model = pickle.load(open(pkl_path, 'rb'))
            logger.info(f"Pickled model loaded successfully: {type(model)}")
            return model
        except Exception as e:
            logger.error(f"Error loading pickled model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_torch_model(pth_path: str):
        """Load PyTorch model"""
        try:
            if gym is None:
                logger.error("gymnasium not available for torch model loading")
                return None

            import torch
            from stable_baselines3.ppo.policies import ActorCriticPolicy

            # Create model instance with the right architecture
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(257,), dtype=np.float32)  # Match saved model
            action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Single output

            # Create policy matching saved architecture
            policy = ActorCriticPolicy(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=lambda _: 0.0,  # Dummy schedule since we're just using for inference
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Match saved 64-unit architecture
            )

            # Load state dict
            state_dict = torch.load(pth_path, map_location='cpu')
            policy.load_state_dict(state_dict)
            policy.eval()  # Set to evaluation mode

            logger.info(f"PyTorch model loaded successfully and reconstructed as ActorCriticPolicy")
            return policy
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_deepseek_model(zip_path: str):
        """Load DeepSeek model from ZIP with custom logic - matches training configuration"""
        try:
            if PPO is None:
                logger.error("stable_baselines3 not available")
                return None

            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading DeepSeek model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            model = PPO.load(zip_path, device=device)
            logger.info(f"DeepSeek model loaded successfully! Policy: {type(model.policy)}")
            return model

        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_claude_model(zip_path: str):
        """Load Claude model from ZIP with custom logic - matches training configuration with RiskAwareExtractor"""
        try:
            if PPO is None:
                logger.error("stable_baselines3 not available")
                return None

            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading Claude model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            model = PPO.load(zip_path, device=device)
            logger.info(f"Claude model loaded successfully! Policy: {type(model.policy)}")
            return model

        except Exception as e:
            logger.error(f"Error loading Claude model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_kimi_model(zip_path: str):
        """Load Kimi model from ZIP with custom logic - matches training configuration with default policy kwargs"""
        try:
            if PPO is None:
                logger.error("stable_baselines3 not available")
                return None

            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading Kimi model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            model = PPO.load(zip_path, device=device)
            logger.info(f"Kimi model loaded successfully! Policy: {type(model.policy)}")
            return model

        except Exception as e:
            logger.error(f"Error loading Kimi model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_gpt_model(zip_path: str):
        """Load GPT model from ZIP with custom logic - matches training configuration from train.py"""
        try:
            from stable_baselines3 import PPO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading GPT model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            model = PPO.load(zip_path, device=device)
            logger.info(f"GPT model loaded successfully! Policy: {type(model.policy)}")
            return model

        except ImportError as e:
            logger.error(f"stable_baselines3 not available for GPT: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading GPT model: {e}", exc_info=True)
            return None

    @staticmethod
    def load_model_by_type(model_path: str):
        """Unified loader that detects model type and loads accordingly"""
        if not ModelLoaders.check_model_file(model_path):
            return None

        # Conditional loading based on model name
        if "gemini.zip" in model_path:
            return ModelLoaders.load_stable_baselines3_model(model_path)
        elif "deepseek.zip" in model_path:
            return ModelLoaders.load_deepseek_model(model_path)
        elif "claude.zip" in model_path:
            return ModelLoaders.load_claude_model(model_path)
        elif "kimi.zip" in model_path:
            return ModelLoaders.load_kimi_model(model_path)
        elif "gpt.zip" in model_path:
            return ModelLoaders.load_gpt_model(model_path)
        elif "grok.zip" in model_path:
            return ModelLoaders.load_stable_baselines3_model(model_path)
        elif model_path.endswith('.zip'):
            return ModelLoaders.load_stable_baselines3_model(model_path)
        elif model_path.endswith('.pkl'):
            return ModelLoaders.load_pickle_model(model_path)
        elif model_path.endswith('.pth'):
            return ModelLoaders.load_torch_model(model_path)
        else:
            logger.error(f"Unsupported model format: {model_path}")
            return None
