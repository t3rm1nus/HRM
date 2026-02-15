"""
Config Loader Module

Carga la configuración inicial desde initial_state.json y proporciona
funciones para acceder a los valores de configuración.

Este módulo centraliza el acceso a la configuración inicial del sistema.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "capital_usdt": 3000.0,
    "btc": 0.0,
    "eth": 0.0,
    "mode": "paper",
    "auto_learning": "fix",
    "initial_balances": {
        "USDT": 3000.0,
        "BTC": 0.0,
        "ETH": 0.0
    },
    "reset_singletons": True
}


def load_initial_state(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Load initial state from JSON file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Ensure all default keys exist
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading {config_path}: {e}")
            print("   Using default configuration")
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file if it doesn't exist
        save_initial_state(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG.copy()


def save_initial_state(config: Dict[str, Any], config_path: str = "initial_state.json") -> bool:
    """
    Save initial state to JSON file.
    
    Args:
        config: Configuration dict to save
        config_path: Path to the initial state JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"⚠️  Error saving {config_path}: {e}")
        return False


def get_initial_balances(config_path: str = "initial_state.json") -> Dict[str, float]:
    """
    Get initial balances for SimulatedExchangeClient.
    
    Returns:
        Dict with BTC, ETH, and USDT balances
    """
    config = load_initial_state(config_path)
    
    # Prefer initial_balances if available
    if "initial_balances" in config:
        balances = config["initial_balances"]
        return {
            "BTC": float(balances.get("BTC", 0.0)),
            "ETH": float(balances.get("ETH", 0.0)),
            "USDT": float(balances.get("USDT", 3000.0))
        }
    
    # Fallback to individual fields
    return {
        "BTC": float(config.get("btc", 0.0)),
        "ETH": float(config.get("eth", 0.0)),
        "USDT": float(config.get("capital_usdt", 3000.0))
    }


def get_capital_usd(config_path: str = "initial_state.json") -> float:
    """Get initial capital in USD."""
    config = load_initial_state(config_path)
    return float(config.get("capital_usdt", 3000.0))


def get_mode(config_path: str = "initial_state.json") -> str:
    """Get trading mode (paper/live)."""
    config = load_initial_state(config_path)
    return config.get("mode", "paper")


def get_auto_learning(config_path: str = "initial_state.json") -> str:
    """Get auto-learning mode."""
    config = load_initial_state(config_path)
    return config.get("auto_learning", "fix")


def should_reset_singletons(config_path: str = "initial_state.json") -> bool:
    """Check if singletons should be reset."""
    config = load_initial_state(config_path)
    return config.get("reset_singletons", True)


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Get cached configuration or load from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_initial_state(config_path)
    
    return _config_cache


def reload_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    _config_cache = load_initial_state(config_path)
    return _config_cache


def reset_config_cache():
    """Reset the configuration cache."""
    global _config_cache
    _config_cache = None


# Convenience properties
@property
def initial_balances() -> Dict[str, float]:
    """Get initial balances (uses cached config)."""
    return get_initial_balances()


@property
def capital_usd() -> float:
    """Get initial capital (uses cached config)."""
    return get_capital_usd()


@property
def mode() -> str:
    """Get trading mode (uses cached config)."""
    return get_mode()


if __name__ == "__main__":
    # Test the module
    print("🔧 Config Loader Test")
    print("=" * 50)
    
    config = load_initial_state()
    print(f"📊 Config loaded: {config}")
    print()
    
    balances = get_initial_balances()
    print(f"💰 Initial balances: {balances}")
    print(f"💵 Capital USD: {get_capital_usd()}")
    print(f"🎯 Mode: {get_mode()}")
    print(f"🤖 Auto-learning: {get_auto_learning()}")
    print(f"🔄 Reset singletons: {should_reset_singletons()}")
