# l2_tactic/safe_model_loader.py
import os
import logging
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    """
    Carga el modelo en un contexto aislado para evitar problemas de pickle
    """
    try:
        # Método 1: Intentar carga normal
        logger.info(f"Intentando carga normal desde {model_path}")
        model = PPO.load(model_path)
        return model
        
    except Exception as e:
        logger.warning(f"Error en carga normal: {e}")
        
        # Método 2: Extraer y cargar manualmente
        try:
            import zipfile
            import tempfile
            import torch
            
            logger.info("Intentando carga manual con extracción...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extraer el ZIP
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Cargar el estado del modelo manualmente
                model = PPO('MlpPolicy', 'CartPole-v1')  # Modelo dummy
                
                # Cargar weights manualmente
                policy_path = os.path.join(temp_dir, 'policy.pth')
                if os.path.exists(policy_path):
                    device = torch.device('cpu')
                    model.policy.load_state_dict(
                        torch.load(policy_path, map_location=device, weights_only=True)
                    )
                    logger.info("✅ Modelo cargado manualmente")
                    return model
                else:
                    raise Exception("Archivo policy.pth no encontrado")
                    
        except Exception as manual_e:
            logger.error(f"Error en carga manual: {manual_e}")
            raise