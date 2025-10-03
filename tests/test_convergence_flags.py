#!/usr/bin/env python3
"""
Test script to verify convergence configuration flags can be loaded from JSON.
"""
import os
import json
from typing import Dict, Any

def load_convergence_config() -> Dict[str, Any]:
    """Load convergence configuration from JSON file."""
    config_file = "core/config/convergence_config.json"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration has all required flags."""
    required_flags = ['trend_following_mode', 'use_mean_reversion', 'ma_short', 'ma_long', 'min_trend_confidence']

    for flag in required_flags:
        if flag not in config:
            print(f"❌ Missing required flag: {flag}")
            return False
        print(f"✅ Found flag {flag}: {config[flag]} (type: {type(config[flag]).__name__})")

    return True

def demonstrate_usage():
    """Demonstrate how to use the configuration flags."""
    try:
        config = load_convergence_config()
        print("🔧 CONVERGENCE CONFIGURATION LOADED:")
        print("=" * 40)

        if validate_config(config):
            print("\n✅ CONFIG VALIDATION PASSED\n")

            # Example usage
            trend_mode = config['trend_following_mode']
            mean_rev = config['use_mean_reversion']
            short_ma = config['ma_short']
            long_ma = config['ma_long']
            min_conf = config['min_trend_confidence']

            print("EXAMPLE USAGE:")
            print("-" * 20)
            print(f"Trend Following: {'ENABLED' if trend_mode else 'DISABLED'}")
            print(f"Mean Reversion: {'ENABLED' if mean_rev else 'DISABLED'}")
            print(f"Short MA Period: {short_ma}")
            print(f"Long MA Period: {long_ma}")
            print(f"Min Trend Confidence: {min_conf}")

            if trend_mode and not mean_rev:
                print("📈 TREND FOLLOWING MODE ACTIVE")
            elif not trend_mode and mean_rev:
                print("📊 MEAN REVERSION MODE ACTIVE")
            else:
                print("🔄 MIXED MODE ACTIVE")
        else:
            print("❌ CONFIG VALIDATION FAILED")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    demonstrate_usage()
