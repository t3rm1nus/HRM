"""
FinRL Integration para L2_tactic - CORREGIDO con gymnasium
=========================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from core.logging import logger
from .models import TacticalSignal

class FinRLProcessor:
    """
    Procesador FinRL REAL - carga modelo con gymnasium
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(str(model_path).replace('\\', '/'))  # Fix Windows paths
        self.model = None
        self.observation_space = None
        self.action_space = None
        self.is_loaded = False
        
        # Cargar el modelo real
        self._load_real_model()
    
    def _load_real_model(self):
        """
        Carga el modelo FinRL real con gymnasium
        """
        try:
            logger.info(f"🤖 Cargando modelo FinRL desde: {self.model_path}")
            
            # Verificar directorio
            if not self.model_path.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {self.model_path}")
            
            # Verificar archivos críticos
            policy_file = self.model_path / "policy.pth"
            if not policy_file.exists():
                raise FileNotFoundError(f"policy.pth no encontrado en {self.model_path}")
            
            logger.info(f"✅ Archivos del modelo encontrados")
            
            # Importar dependencias
            try:
                from stable_baselines3 import PPO
                import gymnasium as gym
                import torch
                logger.info("✅ stable_baselines3 y gymnasium importados")
            except ImportError as e:
                logger.error(f"❌ Error importando dependencias: {e}", exc_info=True)
                raise ImportError(f"Falta dependencia: {e}")
            
            # Cargar policy state
            logger.info("🔧 Cargando modelo desde policy.pth...")
            policy_state = torch.load(str(policy_file), map_location='cpu', weights_only=True)
            logger.info("✅ Estado del modelo cargado desde policy.pth")
            
            # Inferir dimensiones del modelo
            obs_dim, action_dim = self._infer_model_dimensions(policy_state)
            if obs_dim is None or action_dim is None:
                logger.warning("⚠️ No se pudieron inferir dimensiones, usando valores por defecto: obs=28, actions=2")
                obs_dim = 28
                action_dim = 2
            logger.info(f"📊 Dimensiones inferidas: obs={obs_dim}, actions={action_dim}")
            
            # Crear spaces con gymnasium
            from gymnasium import spaces
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(action_dim)
            
            # Crear entorno dummy
            class DummyEnv(gym.Env):
                def __init__(self, obs_dim, action_dim):
                    super().__init__()
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                    )
                    self.action_space = spaces.Discrete(action_dim)
                
                def step(self, action):
                    return np.zeros(self.observation_space.shape), 0, False, False, {}
                
                def reset(self, seed=None, **kwargs):
                    return np.zeros(self.observation_space.shape), {}
            
            dummy_env = DummyEnv(obs_dim, action_dim)
            
            # Configurar arquitectura de red
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[256, 128],  # Policy network
                    vf=[256, 128]   # Value network
                )
            )
            
            # Crear modelo PPO
            try:
                self.model = PPO(
                    'MlpPolicy',
                    env=dummy_env,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device='cpu'
                )
                logger.info("✅ Modelo PPO creado exitosamente")
            except Exception as e:
                logger.error(f"❌ Error creando modelo PPO: {e}", exc_info=True)
                raise
            
            # Cargar el policy state filtrado
            try:
                if 'policy' in policy_state:
                    policy_dict = policy_state['policy']
                else:
                    policy_dict = policy_state
                filtered_policy_dict = {k: v for k, v in policy_dict.items() if k != 'log_std'}
                self.model.policy.load_state_dict(filtered_policy_dict, strict=False)
                logger.info("✅ Modelo cargado manualmente desde policy.pth (con keys filtradas)")
            except Exception as e:
                logger.error(f"❌ Error cargando estado del modelo: {e}", exc_info=True)
                self.model = None
                raise
            
            # Verificar inicialización
            if self.model is None or not hasattr(self.model, 'predict'):
                logger.error("❌ Modelo PPO no inicializado o no tiene método predict")
                raise ValueError("Modelo PPO no inicializado")
            
            self.is_loaded = True
            logger.info("✅ Modelo FinRL REAL cargado exitosamente")
            logger.info(f"📊 Observation space: {self.observation_space}")
            logger.info(f"🎯 Action space: {self.action_space}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo real: {e}", exc_info=True)
            logger.error("❌ MODELO REAL NO SE PUDO CARGAR")
            self.model = None
            self.is_loaded = False
            raise Exception(f"Falló carga del modelo real: {e}")
    def _infer_model_dimensions(self, policy_state):
        """
        Infiere dimensiones del modelo desde el state dict
        """
        try:
            obs_dim = None
            action_dim = None
            
            # Buscar en el diccionario del policy
            policy_dict = policy_state.get('policy', policy_state)
            
            # Buscar dimensión de observación
            for key, tensor in policy_dict.items():
                if 'mlp_extractor' in key and 'weight' in key and tensor.dim() == 2:
                    obs_dim = tensor.shape[1]  # Input dimension
                    logger.debug(f"Obs dim desde {key}: {obs_dim}")
                    break
            
            # Buscar dimensión de acción
            for key, tensor in policy_dict.items():
                if 'action_net' in key and 'weight' in key:
                    action_dim = tensor.shape[0]  # Output dimension
                    logger.debug(f"Action dim desde {key}: {action_dim}")
                    break
            
            # Fallback: buscar en otras capas
            if obs_dim is None:
                for key, tensor in policy_dict.items():
                    if 'weight' in key and tensor.dim() == 2:
                        obs_dim = tensor.shape[1]
                        logger.debug(f"Obs dim fallback desde {key}: {obs_dim}")
                        break
            
            if action_dim is None:
                # Asumir 3 acciones por defecto (buy, hold, sell)
                action_dim = 3
                logger.debug(f"Action dim por defecto: {action_dim}")
            
            return obs_dim, action_dim
            
        except Exception as e:
            logger.error(f"Error inferiendo dimensiones: {e}")
            return None, None
    
    # En finrl_integration.py, reemplazar el método generate_signal:

    def generate_signal(self, market_data: dict, symbol: str) -> Optional[TacticalSignal]:
        if not self.is_loaded or self.model is None:
            logger.error(f"❌ Modelo no cargado para generar señal de {symbol}")
            return None
        
        try:
            # Corregir: acceder directamente a market_data, no market_data[symbol]
            obs = self._prepare_observation(market_data, symbol)
            if obs is None:
                logger.warning(f"⚠️ Observación no válida para {symbol}")
                return None
            
            action, _states = self.model.predict(obs, deterministic=True)
            logger.debug(f"Action raw: {action}, type: {type(action)}, shape: {getattr(action, 'shape', 'no shape')}")
            
            side = 'buy' if action == 0 else 'sell'
            
            # Corregir acceso a indicators
            indicators = market_data.get('indicators', {})
            
            return TacticalSignal(
                symbol=symbol,
                strength=0.7,
                confidence=0.85,
                side=side,
                features=indicators,  # Cambiar de market_data[symbol] a indicators
                timestamp=pd.Timestamp.now(),  # Usar pd.Timestamp directamente
                signal_type='finrl',
                metadata={'model': 'finrl', 'action': int(action)}
            )
        except Exception as e:
            logger.error(f"❌ Error generando señal FinRL: {e}")
            return None 
    # En finrl_integration.py, reemplazar el método _prepare_observation:

    def _prepare_observation(self, market_data: Dict[str, Any], symbol: str) -> Optional[np.ndarray]:
        """
        Prepara observación para el modelo - CORREGIDO
        """
        try:
            # market_data ahora viene directamente como los datos del símbolo
            ohlcv = market_data.get('ohlcv', {})
            indicators = market_data.get('indicators', {})
            
            if not ohlcv:
                logger.warning(f"No hay datos OHLCV para {symbol}")
                return None
            
            # Features básicas
            close = float(ohlcv.get('close', 0))
            volume = float(ohlcv.get('volume', 0))
            high = float(ohlcv.get('high', close))
            low = float(ohlcv.get('low', close))
            
            # Evitar división por cero
            close_safe = max(close, 0.001)
            
            features = [
                close / 50000,  # Precio normalizado (ajustar según tu rango)
                volume / 1000,  # Volumen normalizado
                high / close_safe,  # High/Close ratio
                low / close_safe,   # Low/Close ratio
                indicators.get('rsi', 50) / 100,
                indicators.get('macd', 0) / 100,  # Normalizar MACD
                indicators.get('macd_signal', 0) / 100,
                indicators.get('bb_upper', close) / close_safe,
                indicators.get('bb_lower', close) / close_safe,
                indicators.get('sma_20', close) / close_safe,
                indicators.get('ema_12', close) / close_safe,
                market_data.get('change_24h', 0)  # Ya debería estar normalizado
            ]
            
            # Agregar features adicionales si el modelo las necesita
            additional_features = [
                indicators.get('volatility', 0),
                indicators.get('vol_ratio', 1.0),
                indicators.get('sma_10', close) / close_safe,
                indicators.get('ema_10', close) / close_safe
            ]
            features.extend(additional_features)
            
            # Ajustar longitud según el modelo
            if hasattr(self.observation_space, 'shape'):
                expected_len = self.observation_space.shape[0]
                if len(features) < expected_len:
                    features.extend([0.0] * (expected_len - len(features)))
                elif len(features) > expected_len:
                    features = features[:expected_len]
            
            obs_array = np.array(features, dtype=np.float32)
            
            # Limpiar valores inválidos
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values
            obs_array = np.clip(obs_array, -100, 100)
            
            logger.debug(f"Observación preparada para {symbol}: shape={obs_array.shape}, "
                        f"range=[{obs_array.min():.3f}, {obs_array.max():.3f}]")
            
            return obs_array
            
        except Exception as e:
            logger.error(f"Error preparando observación para {symbol}: {e}")
            return None
    def _action_to_signal(self, action: int, symbol: str, market_data: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Convierte acción a señal
        """
        try:
            # Tu modelo tiene 2 acciones: 0=Buy, 1=Sell (sin Hold)
            action_map = {0: 'buy', 1: 'sell'}
            
            if action not in action_map:
                logger.warning(f"Acción desconocida: {action}, esperadas: {list(action_map.keys())}")
                return None
            
            side = action_map[action]
            
            # Calcular strength para modelo de 2 acciones
            base_strength = 0.7
            volume_factor = min(market_data.get('ohlcv', {}).get('volume', 0) / 10000, 0.2)
            
            if action == 0:  # Buy
                strength = base_strength + volume_factor
            else:  # Sell (action == 1)
                strength = -(base_strength + volume_factor)
            
            return TacticalSignal(
                symbol=symbol,
                signal_type='finrl_ppo_real',
                strength=strength,
                confidence=0.85,
                side=side,
                features=market_data,
                timestamp=pd.Timestamp.now().timestamp(),
                metadata={
                    'model': 'FinRL_PPO_REAL',
                    'action': int(action),
                    'model_loaded': True
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error convirtiendo acción: {e}")
            return None