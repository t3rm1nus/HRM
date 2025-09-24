# train_grok_ultra_optimized.py
# Versión ULTRA-OPTIMIZADA para máxima velocidad de entrenamiento
# Correcciones principales: Environment simplificado, eliminación de overheads, FIX activation_fn

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
import warnings
import numpy as np
import pandas as pd
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import time

# CONFIGURACIÓN ULTRA-OPTIMIZADA
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = Path("C:/proyectos/charniRich/envs/grok/train/data/processed/normalized_grok.parquet")
MODELS_DIR = BASE_DIR / "models" / "L2" / "ai_model_data_multiasset"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
L3_SIM_PATH = BASE_DIR / "data" / "datos_inferencia" / "l3_output.json"

# CONFIGURACIÓN OPTIMIZADA PARA VELOCIDAD Y APRENDIZAJE
BATCH_SIZE = 256  # Aumentado para mejor gradientes
LEARNING_RATE_MAIN = 3e-4  # Aumentado para aprendizaje más rápido
N_STEPS = 2048  # Aumentado para mejor exploration
N_EPOCHS_MAIN = 10  # Aumentado para mejor utilización de datos
N_EPOCHS_FINE = 5
GAMMA = 0.995  # Más enfocado en recompensas a largo plazo
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2  # Menos conservador para permitir más cambios
ENT_COEF = 0.01  # Aumentado para más exploración inicial
VF_COEF = 0.5  # Reducido para balancear con policy loss
MAX_GRAD_NORM = 0.5  # Menos restrictivo
N_ENVS = 1
DEVICE = get_device("auto")
TRANSACTION_COST = 0.001
MAX_EXPOSURE = {"BTCUSDT": 0.20, "ETHUSDT": 0.15}
WINDOW_LOOKBACK = 15  # Reducido
OBS_CLIP = 5.0  # Reducido
REWARD_CLIP = 5.0  # Reducido

print(f"Using device: {DEVICE}")


class FastTradingEnv(gym.Env):
    """Environment ultra-optimizado - elimina todos los overheads"""

    def __init__(self, df_dict, initial_balance=10000):
        super().__init__()

        # Configuración básica
        self.tickers = list(df_dict.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance

        # PRE-COMPUTAR TODO AL INICIALIZAR
        self._precompute_all_data(df_dict)

        # Spaces simplificados
        obs_size = self.n_assets * 4 + 5  # returns, volumes, rsi, positions + portfolio info
        self.observation_space = spaces.Box(low=-OBS_CLIP, high=OBS_CLIP, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Estado inicial
        self.reset()

    def _precompute_all_data(self, df_dict):
        """Pre-computar TODOS los datos para eliminar cálculos en runtime"""
        # Encontrar longitud mínima
        min_len = min(len(df_dict[t]) for t in self.tickers)
        self.max_steps = min_len - 2

        print(f"Pre-computando {self.max_steps} steps para {self.n_assets} assets...")

        # Arrays pre-computados - TODO en memoria
        self.prices = np.zeros((self.max_steps, self.n_assets), dtype=np.float32)
        self.returns = np.zeros((self.max_steps, self.n_assets), dtype=np.float32)
        self.volumes = np.zeros((self.max_steps, self.n_assets), dtype=np.float32)
        self.rsi = np.zeros((self.max_steps, self.n_assets), dtype=np.float32)

        # Procesar cada ticker
        for i, ticker in enumerate(self.tickers):
            df = df_dict[ticker].iloc[:self.max_steps].copy()

            # Limpiar datos de manera vectorizada
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['close'] = df['close'].fillna(method='ffill').fillna(1.0)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(1.0)
            df['rsi'] = pd.to_numeric(df['rsi'], errors='coerce').fillna(50.0)

            # Guardar en arrays
            self.prices[:, i] = df['close'].values

            # Calcular returns
            price_series = df['close'].values
            returns_raw = np.diff(price_series) / price_series[:-1]
            self.returns[1:, i] = np.nan_to_num(returns_raw, 0.0)
            self.returns[0, i] = 0.0

            # Volume normalizado (log transform)
            volume_raw = np.log1p(df['volume'].values)
            volume_norm = (volume_raw - np.mean(volume_raw)) / (np.std(volume_raw) + 1e-8)
            self.volumes[:, i] = np.clip(volume_norm, -3, 3)

            # RSI normalizado
            rsi_norm = (df['rsi'].values - 50.0) / 50.0
            self.rsi[:, i] = np.clip(rsi_norm, -1, 1)

        # Clip todos los arrays
        self.returns = np.clip(self.returns, -0.2, 0.2)  # Max 20% return per step

        print(f"✅ Pre-computación completada:")
        print(f"  - Prices shape: {self.prices.shape}")
        print(f"  - Returns range: [{self.returns.min():.4f}, {self.returns.max():.4f}]")
        print(f"  - Volume range: [{self.volumes.min():.4f}, {self.volumes.max():.4f}]")
        print(f"  - RSI range: [{self.rsi.min():.4f}, {self.rsi.max():.4f}]")

    def reset(self, seed=None, options=None):
        """Reset ultra-rápido"""
        self.step_idx = 0
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_history = [self.initial_balance]
        # Reset action tracking for consistency reward
        if hasattr(self, 'prev_action'):
            delattr(self, 'prev_action')
        return self._get_obs(), {}

    def _get_obs(self):
        """Observación ultra-simplificada - CERO cálculos complejos"""
        # Boundary check
        step = min(self.step_idx, self.max_steps - 1)

        # Datos pre-computados (acceso directo a arrays)
        current_returns = self.returns[step]
        current_volumes = self.volumes[step]
        current_rsi = self.rsi[step]
        current_prices = self.prices[step]

        # Portfolio info simple
        portfolio_value = self.balance + np.sum(self.positions * current_prices)

        # Posiciones normalizadas
        pos_values = self.positions * current_prices
        pos_norm = pos_values / max(portfolio_value, 1.0)

        # Performance simple
        performance = (portfolio_value / self.initial_balance) - 1.0

        # Observación compacta
        obs = np.concatenate([
            current_returns,  # Returns de cada asset
            current_volumes,  # Volume normalizado
            current_rsi,  # RSI normalizado
            pos_norm,  # Posiciones normalizadas
            [performance,  # Performance total
             self.balance / portfolio_value if portfolio_value > 0 else 1.0,  # Cash ratio
             np.sum(np.abs(pos_norm)),  # Total exposure
             len(self.portfolio_history) / self.max_steps,  # Progress
             np.std(self.portfolio_history[-5:]) if len(self.portfolio_history) > 4 else 0.0]  # Recent volatility
        ]).astype(np.float32)

        # Clipping final
        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    def step(self, action):
        """Step ultra-optimizado"""
        # Check termination
        if self.step_idx >= self.max_steps - 1:
            return self._get_obs(), 0.0, True, True, {"portfolio_value": self._get_portfolio_value()}

        # Execute action (simplificado)
        reward = self._execute_action_fast(action)

        # Avanzar step
        self.step_idx += 1

        # Return
        obs = self._get_obs()
        done = self.step_idx >= self.max_steps - 1

        return obs, reward, done, False, {"portfolio_value": self._get_portfolio_value()}

    def _execute_action_fast(self, action):
        """Ejecución de acción ultra-optimizada con mejor reward engineering"""
        # Precios actuales (acceso directo)
        step = min(self.step_idx, self.max_steps - 1)
        current_prices = self.prices[step]

        # Store previous portfolio value for proper reward calculation
        prev_portfolio_value = self.balance + np.sum(self.positions * current_prices)

        # Target weights (acción directa, sin complicaciones)
        action_clipped = np.clip(action, -1, 1)
        max_exposures = np.array([MAX_EXPOSURE.get(ticker, 0.2) for ticker in self.tickers])
        target_weights = action_clipped * max_exposures * 0.5  # Max 50% of limit per action

        # Target positions
        target_values = target_weights * prev_portfolio_value
        target_positions = np.where(current_prices > 0, target_values / current_prices, 0)

        # Calculate trades
        position_changes = target_positions - self.positions
        trade_values = np.abs(position_changes * current_prices)
        total_trade_cost = np.sum(trade_values) * TRANSACTION_COST

        # Update positions
        self.positions = target_positions

        # Update balance (subtract net trades + costs)
        net_trade_value = np.sum(position_changes * current_prices)
        self.balance -= net_trade_value + total_trade_cost

        # Move to next step to get market movement
        next_step = min(self.step_idx + 1, self.max_steps - 1)
        next_prices = self.prices[next_step]

        # Calculate new portfolio value with updated prices
        new_portfolio_value = self.balance + np.sum(self.positions * next_prices)
        self.portfolio_history.append(new_portfolio_value)

        # IMPROVED REWARD CALCULATION
        # 1. Portfolio return (scaled for significance)
        if prev_portfolio_value > 0:
            portfolio_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            base_reward = portfolio_return * 100  # Scale up to make more significant
        else:
            portfolio_return = 0.0
            base_reward = -1.0  # Penalty for bankruptcy

        # 2. Market-relative performance
        if step < self.max_steps - 1:
            market_returns = self.returns[next_step]
            avg_market_return = np.mean(market_returns)
            outperformance = portfolio_return - avg_market_return
            alpha_reward = outperformance * 50  # Reward for beating market
        else:
            alpha_reward = 0.0

        # 3. Position diversity reward
        position_values = np.abs(self.positions * next_prices)
        total_position_value = np.sum(position_values)
        if total_position_value > 0:
            weights = position_values / total_position_value
            # Reward balanced positions, penalize concentration
            diversity = 1.0 - np.sum(weights ** 2)  # Herfindahl index
            diversity_reward = diversity * 2.0
        else:
            diversity_reward = -0.5  # Penalty for no positions

        # 4. Action consistency (reduce random actions)
        if hasattr(self, 'prev_action'):
            action_consistency = 1.0 - np.mean(np.abs(action_clipped - self.prev_action))
            consistency_reward = action_consistency * 0.5
        else:
            consistency_reward = 0.0
        self.prev_action = action_clipped.copy()

        # 5. Risk penalties
        trade_cost_penalty = total_trade_cost / max(prev_portfolio_value, 1.0) * 10

        # Volatility penalty (only if excessive)
        if len(self.portfolio_history) > 5:
            recent_returns = np.diff(self.portfolio_history[-5:]) / np.array(self.portfolio_history[-5:-1])
            volatility = np.std(recent_returns)
            volatility_penalty = max(0, volatility - 0.02) * 10  # Penalize vol > 2%
        else:
            volatility_penalty = 0.0

        # Final reward with better scaling
        reward = (base_reward +
                  alpha_reward +
                  diversity_reward +
                  consistency_reward -
                  trade_cost_penalty -
                  volatility_penalty)

        return np.clip(reward, -REWARD_CLIP, REWARD_CLIP)

    def _get_portfolio_value(self):
        """Cálculo ultra-rápido del valor del portafolio"""
        step = min(self.step_idx, self.max_steps - 1)
        current_prices = self.prices[step]
        return self.balance + np.sum(self.positions * current_prices)


def make_fast_env(df_dict, seed=0):
    """Factory function optimizada - SIN Monitor para reducir overhead"""

    def _init():
        env = FastTradingEnv(df_dict)
        # NO usar Monitor para máxima velocidad
        return env

    return _init


def load_and_split_data():
    """Carga de datos optimizada"""
    print("Cargando datos...")
    start_time = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {DATA_PATH}")

    # Carga con chunks para manejar archivos grandes
    df = pd.read_parquet(DATA_PATH)
    print(f"Datos cargados: {len(df)} registros")

    # Procesar tickers
    df['ticker'] = df['ticker'].apply(lambda x: x + 'USDT' if x in ['BTC', 'ETH'] else x)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Validación rápida
    required_cols = ['date', 'ticker', 'close', 'rsi', 'volume']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Limpieza vectorizada
    numeric_cols = ['close', 'rsi', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(method='ffill').fillna(1.0)
        df[col] = df[col].replace([np.inf, -np.inf], 1.0)

    # Split temporal 80/20 (más datos para validación)
    split_date = df['date'].quantile(0.8)
    print(f"Split temporal en: {split_date}")

    df_train = df[df['date'] < split_date].copy()
    df_val = df[df['date'] >= split_date].copy()

    # Crear diccionarios
    train_dict = {}
    val_dict = {}

    for ticker in df['ticker'].unique():
        train_ticker_data = df_train[df_train['ticker'] == ticker].reset_index(drop=True)
        val_ticker_data = df_val[df_val['ticker'] == ticker].reset_index(drop=True)

        if len(train_ticker_data) > 100 and len(val_ticker_data) > 50:  # Mínimo de datos
            train_dict[ticker] = train_ticker_data
            val_dict[ticker] = val_ticker_data

    print(f"📊 DATOS PROCESADOS:")
    print(f"  • Entrenamiento: {sum(len(df) for df in train_dict.values()):,} registros")
    print(f"  • Validación: {sum(len(df) for df in val_dict.values()):,} registros")
    print(f"  • Tickers: {list(train_dict.keys())}")
    print(f"  • Carga completada en: {time.time() - start_time:.2f}s")

    return train_dict, val_dict


def train_model_fast(df_train_dict, df_val_dict):
    """Entrenamiento ultra-optimizado"""
    print("🚀 INICIANDO ENTRENAMIENTO ULTRA-OPTIMIZADO")
    start_time = time.time()

    # Environments SIN VecNormalize (elimina overhead significativo)
    print("Creando environments...")
    train_env = DummyVecEnv([make_fast_env(df_train_dict)])
    val_env = DummyVecEnv([make_fast_env(df_val_dict)])

    # Modelo optimizado - FIX: usar torch.nn.Tanh en lugar de string
    print("Creando modelo PPO optimizado...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE_MAIN,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS_MAIN,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        device=DEVICE,
        tensorboard_log=str(MODELS_DIR / "tb_logs"),
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},  # Red más grande para mejor aprendizaje
            "activation_fn": torch.nn.Tanh,  # FIX: usar torch.nn.Tanh en lugar de string
            "ortho_init": False,  # Desactivar inicialización ortogonal
            "log_std_init": -0.5  # Inicializar con menor varianza para actions más focalizadas
        }
    )

    # Cargar checkpoint si existe
    checkpoint_files = [
        "ppo_hrm_100000_steps.zip",
        "ppo_hrm_75000_steps.zip",
        "ppo_hrm_50000_steps.zip",
        "ppo_hrm_25000_steps.zip"
    ]

    model_loaded = False
    loaded_steps = 0

    for checkpoint_file in checkpoint_files:
        checkpoint_path = MODELS_DIR / "checkpoints" / checkpoint_file
        if checkpoint_path.exists():
            print(f"📁 Intentando cargar: {checkpoint_path}")
            try:
                model = PPO.load(str(checkpoint_path), env=train_env)
                loaded_steps = int(checkpoint_file.split('_')[-2])
                print(f"✅ Cargado desde {loaded_steps:,} steps")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠ Error cargando {checkpoint_file}: {e}")
                continue

    if not model_loaded:
        print("🆕 Iniciando entrenamiento desde cero")
        loaded_steps = 0

    # Callbacks mínimos y más robustos
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(MODELS_DIR / "checkpoints"),
        log_path=str(MODELS_DIR / "logs"),
        eval_freq=20000,  # Menos frecuente para evitar bloqueos
        n_eval_episodes=3,  # Menos episodios para evaluación más rápida
        deterministic=True,
        render=False,
        verbose=0
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Menos frecuente para evitar I/O blocks
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="ppo_hrm"
    )

    # ENTRENAMIENTO RÁPIDO CON MEJOR MANEJO DE MEMORIA
    target_steps = 100000  # Reducido para completar más rápido
    remaining_steps = max(25000, target_steps - loaded_steps)

    print(f"📈 PLAN DE ENTRENAMIENTO:")
    print(f"  • Steps completados: {loaded_steps:,}")
    print(f"  • Steps objetivo: {remaining_steps:,}")
    print(f"  • Meta total: {target_steps:,}")

    # Fase 1: Entrenamiento principal con manejo de memoria
    print("=== FASE 1: ENTRENAMIENTO PRINCIPAL ===")
    training_start = time.time()

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[checkpoint_callback],  # Solo checkpoint, eval por separado
            progress_bar=True,
            reset_num_timesteps=False
        )

        # Evaluación manual después del entrenamiento (más seguro)
        print("🔍 Evaluación manual...")
        mean_reward, std_reward = evaluate_policy(
            model, val_env,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        print(f"Evaluación completada: {mean_reward:.4f} ± {std_reward:.4f}")

    except Exception as e:
        print(f"⚠️ Error durante entrenamiento: {e}")
        mean_reward, std_reward = 0.0, 0.0

    phase1_time = time.time() - training_start
    actual_fps = remaining_steps / phase1_time
    print(f"⚡ Fase 1 completada - FPS promedio: {actual_fps:.0f}")

    # Fase 2: Fine-tuning (solo si el FPS es bueno)
    if actual_fps > 50:
        print("=== FASE 2: FINE-TUNING ===")
        try:
            # Reducir learning rate para fine-tuning
            original_lr = model.learning_rate
            model.learning_rate = original_lr * 0.3

            model.learn(
                total_timesteps=15000,  # Reducido
                progress_bar=True,
                reset_num_timesteps=False
            )

            # Restore original learning rate
            model.learning_rate = original_lr

        except Exception as e:
            print(f"⚠️ Error en fine-tuning: {e}")
    else:
        print("⚠️ FPS bajo, saltando fine-tuning")

    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = MODELS_DIR / f"ppo_hrm_final_{timestamp}.zip"
    model.save(str(model_path))

    best_model_path = MODELS_DIR / "ppo_hrm_best.zip"
    model.save(str(best_model_path))

    # Evaluación final (con mejor manejo de errores)
    print("🔍 Evaluación final...")
    try:
        final_mean_reward, final_std_reward = evaluate_policy(
            model, val_env,
            n_eval_episodes=10,
            deterministic=True
        )
    except Exception as e:
        print(f"⚠️ Error en evaluación final: {e}")
        final_mean_reward, final_std_reward = mean_reward, std_reward

    total_time = time.time() - start_time
    total_steps_trained = remaining_steps + (15000 if actual_fps > 50 else 0)
    overall_fps = total_steps_trained / max(time.time() - training_start, 1)

    # Resumen
    print(f"\n📊 RESUMEN FINAL:")
    print(f"  • Tiempo total: {total_time / 3600:.2f} horas")
    print(f"  • FPS promedio: {overall_fps:.0f}")
    print(f"  • Mean reward: {final_mean_reward:.4f}")
    print(f"  • Std reward: {final_std_reward:.4f}")
    print(f"  • Modelo guardado: {best_model_path}")

    # Guardar métricas
    from l2_tactic.utils import safe_float
    metrics = {
        "training_completed": datetime.now().isoformat(),
        "total_time_hours": total_time / 3600,
        "fps_average": safe_float(overall_fps),
        "initial_steps": loaded_steps,
        "final_evaluation": {
            "mean_reward": safe_float(final_mean_reward),
            "std_reward": safe_float(final_std_reward)
        },
        "model_path": str(best_model_path)
    }

    metrics_path = MODELS_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    train_env.close()
    val_env.close()

    return model, overall_fps


if __name__ == "__main__":
    print("=== 🚀 ENTRENAMIENTO GROK ULTRA-OPTIMIZADO ===")

    try:
        # Cargar datos
        df_train_dict, df_val_dict = load_and_split_data()

        # Verificar que tenemos datos
        if not df_train_dict or not df_val_dict:
            raise ValueError("No hay suficientes datos para entrenar")

        # Entrenamiento
        model, fps = train_model_fast(df_train_dict, df_val_dict)

        # Generar L3 sample
        sample_l3 = {
            "regime": "neutral",
            "asset_allocation": {"weight": 0.5},
            "risk_appetite": "conservative",
            "sentiment_score": 0.6,
            "volatility_forecast": {"BTCUSDT": 0.8, "ETHUSDT": 0.8},
            "timestamp": datetime.now().isoformat()
        }

        L3_SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(L3_SIM_PATH, 'w') as f:
            json.dump(sample_l3, f, indent=2)

        print(f"✅ L3 sample guardado: {L3_SIM_PATH}")

        # Mensaje final
        if fps > 100:
            print("🎉 ENTRENAMIENTO COMPLETADO CON ÉXITO")
            print(f"Velocidad excelente: {fps:.0f} FPS")
        elif fps > 50:
            print("✅ ENTRENAMIENTO COMPLETADO")
            print(f"Velocidad aceptable: {fps:.0f} FPS")
        else:
            print("⚠️ ENTRENAMIENTO COMPLETADO (velocidad baja)")
            print(f"Velocidad: {fps:.0f} FPS - considerar optimizaciones adicionales")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
