"""
Configuration for convergence and technical strength sizing enhancements.
Provides gradual rollout controls and safety settings.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from core.logging import logger


class ConvergenceConfig:
    """
    Configuration manager for convergence and technical strength features.
    Supports gradual rollout and safety controls.
    """

    def __init__(self, config_file: str = "convergence_config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"📂 Convergence config loaded from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"❌ Error loading convergence config: {e}")
                return self._get_default_config()
        else:
            logger.info("📄 Convergence config file not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with conservative settings"""
        return {
            "version": "1.0",
            "enabled": False,  # Start disabled for safety
            "rollout_phase": "disabled",
            "safety_mode": "conservative",
            "features": {
                "convergence_multiplier": {
                    "enabled": False,
                    "max_multiplier": 1.5,
                    "min_multiplier": 0.5
                },
                "technical_strength_scoring": {
                    "enabled": False,
                    "validation_enabled": False,
                    "bonus_enabled": False
                },
                "circuit_breakers": {
                    "enabled": True,
                    "max_position_size_usd": 100000,
                    "min_technical_strength": 0.1,
                    "reject_on_error": True
                }
            },
            "rollout_schedule": {
                "phase_1": {
                    "name": "monitoring_only",
                    "enabled": False,
                    "description": "Log calculations but don't apply them"
                },
                "phase_2": {
                    "name": "conservative_enabled",
                    "enabled": False,
                    "description": "Enable with conservative limits"
                },
                "phase_3": {
                    "name": "moderate_enabled",
                    "enabled": False,
                    "description": "Enable with moderate limits"
                },
                "phase_4": {
                    "name": "full_enabled",
                    "enabled": False,
                    "description": "Full feature enablement"
                }
            },
            "risk_limits": {
                "max_portfolio_allocation": 0.8,  # 80% max allocation
                "min_position_size_usd": 10.0,
                "max_position_size_usd": 50000.0,
                "emergency_stop_threshold": 0.05  # Stop if strength < 5%
            },
            "logging": {
                "detailed_logging": True,
                "alert_on_rejections": True,
                "performance_tracking": True
            },
            "last_updated": datetime.utcnow().isoformat(),
            "created": datetime.utcnow().isoformat()
        }

    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config["last_updated"] = datetime.utcnow().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, default=str)
            logger.info(f"💾 Convergence config saved to {self.config_file}")
        except Exception as e:
            logger.error(f"❌ Error saving convergence config: {e}")

    def is_enabled(self) -> bool:
        """Check if convergence features are enabled"""
        return self.config.get("enabled", False)

    def get_rollout_phase(self) -> str:
        """Get current rollout phase"""
        return self.config.get("rollout_phase", "disabled")

    def get_safety_mode(self) -> str:
        """Get current safety mode"""
        return self.config.get("safety_mode", "conservative")

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if specific feature is enabled"""
        features = self.config.get("features", {})
        return features.get(feature, {}).get("enabled", False)

    def get_feature_config(self, feature: str) -> Dict[str, Any]:
        """Get configuration for specific feature"""
        features = self.config.get("features", {})
        return features.get(feature, {})

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limits configuration"""
        return self.config.get("risk_limits", {})

    def enable_monitoring_only(self):
        """Enable monitoring-only mode (Phase 1)"""
        logger.info("🔄 ENABLING CONVERGENCE FEATURES - PHASE 1: MONITORING ONLY")

        self.config["enabled"] = True
        self.config["rollout_phase"] = "monitoring_only"
        self.config["safety_mode"] = "conservative"

        # Enable logging but not actual application
        self.config["rollout_schedule"]["phase_1"]["enabled"] = True
        self.config["logging"]["detailed_logging"] = True
        self.config["logging"]["performance_tracking"] = True

        # Keep all features disabled for actual application
        self.config["features"]["convergence_multiplier"]["enabled"] = False
        self.config["features"]["technical_strength_scoring"]["enabled"] = False
        self.config["features"]["technical_strength_scoring"]["validation_enabled"] = False
        self.config["features"]["technical_strength_scoring"]["bonus_enabled"] = False

        self.save_config()
        logger.info("✅ Monitoring-only mode enabled. Features will be calculated but not applied.")

    def enable_conservative_mode(self):
        """Enable conservative operational mode (Phase 2)"""
        logger.info("🔄 ENABLING CONVERGENCE FEATURES - PHASE 2: CONSERVATIVE MODE")

        self.config["enabled"] = True
        self.config["rollout_phase"] = "conservative_enabled"
        self.config["safety_mode"] = "conservative"

        # Enable basic features with conservative limits
        self.config["rollout_schedule"]["phase_2"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["max_multiplier"] = 1.3
        self.config["features"]["technical_strength_scoring"]["enabled"] = True
        self.config["features"]["technical_strength_scoring"]["validation_enabled"] = True
        self.config["features"]["technical_strength_scoring"]["bonus_enabled"] = False

        # Conservative risk limits
        self.config["risk_limits"]["max_portfolio_allocation"] = 0.6
        self.config["risk_limits"]["max_position_size_usd"] = 25000.0

        self.save_config()
        logger.info("✅ Conservative mode enabled. Basic convergence and validation active with tight limits.")

    def enable_moderate_mode(self):
        """Enable moderate operational mode (Phase 3)"""
        logger.info("🔄 ENABLING CONVERGENCE FEATURES - PHASE 3: MODERATE MODE")

        self.config["enabled"] = True
        self.config["rollout_phase"] = "moderate_enabled"
        self.config["safety_mode"] = "moderate"

        # Enable more features with moderate limits
        self.config["rollout_schedule"]["phase_3"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["max_multiplier"] = 1.8
        self.config["features"]["technical_strength_scoring"]["enabled"] = True
        self.config["features"]["technical_strength_scoring"]["validation_enabled"] = True
        self.config["features"]["technical_strength_scoring"]["bonus_enabled"] = True

        # Moderate risk limits
        self.config["risk_limits"]["max_portfolio_allocation"] = 0.7
        self.config["risk_limits"]["max_position_size_usd"] = 35000.0

        self.save_config()
        logger.info("✅ Moderate mode enabled. Full convergence and technical strength features active.")

    def enable_full_mode(self):
        """Enable full operational mode (Phase 4)"""
        logger.info("🔄 ENABLING CONVERGENCE FEATURES - PHASE 4: FULL MODE")

        self.config["enabled"] = True
        self.config["rollout_phase"] = "full_enabled"
        self.config["safety_mode"] = "aggressive"

        # Enable all features with full limits
        self.config["rollout_schedule"]["phase_4"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["enabled"] = True
        self.config["features"]["convergence_multiplier"]["max_multiplier"] = 2.0
        self.config["features"]["technical_strength_scoring"]["enabled"] = True
        self.config["features"]["technical_strength_scoring"]["validation_enabled"] = True
        self.config["features"]["technical_strength_scoring"]["bonus_enabled"] = True

        # Full risk limits
        self.config["risk_limits"]["max_portfolio_allocation"] = 0.8
        self.config["risk_limits"]["max_position_size_usd"] = 50000.0

        self.save_config()
        logger.info("✅ Full mode enabled. All convergence and technical strength features active with maximum limits.")

    def emergency_disable(self):
        """Emergency disable all features"""
        logger.warning("🚨 EMERGENCY DISABLE: Disabling all convergence features")

        self.config["enabled"] = False
        self.config["rollout_phase"] = "emergency_disabled"
        self.config["safety_mode"] = "emergency"

        # Disable all features
        for feature_name in self.config["features"]:
            self.config["features"][feature_name]["enabled"] = False

        # Reset rollout phases
        for phase in self.config["rollout_schedule"]:
            self.config["rollout_schedule"][phase]["enabled"] = False

        self.save_config()
        logger.warning("✅ All convergence features disabled for safety")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        return {
            "enabled": self.is_enabled(),
            "rollout_phase": self.get_rollout_phase(),
            "safety_mode": self.get_safety_mode(),
            "features_status": {
                feature: self.is_feature_enabled(feature)
                for feature in self.config.get("features", {})
            },
            "risk_limits": self.get_risk_limits(),
            "last_updated": self.config.get("last_updated"),
            "config_file": self.config_file
        }

    def validate_config(self) -> bool:
        """Validate configuration integrity"""
        try:
            # Check required fields
            required_fields = ["version", "enabled", "features", "risk_limits"]
            for field in required_fields:
                if field not in self.config:
                    logger.error(f"❌ Missing required config field: {field}")
                    return False

            # Validate risk limits
            risk_limits = self.config.get("risk_limits", {})
            if risk_limits.get("max_portfolio_allocation", 0) > 1.0:
                logger.error("❌ Invalid max_portfolio_allocation: must be <= 1.0")
                return False

            if risk_limits.get("min_position_size_usd", 0) < 0:
                logger.error("❌ Invalid min_position_size_usd: must be >= 0")
                return False

            logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False


# Global configuration instance
_convergence_config = None

def get_convergence_config() -> ConvergenceConfig:
    """Get global convergence configuration instance"""
    global _convergence_config
    if _convergence_config is None:
        _convergence_config = ConvergenceConfig()
    return _convergence_config


def enable_convergence_features(phase: str = "monitoring_only"):
    """
    Convenience function to enable convergence features at specified phase

    Args:
        phase: "monitoring_only", "conservative", "moderate", "full"
    """
    config = get_convergence_config()

    if phase == "monitoring_only":
        config.enable_monitoring_only()
    elif phase == "conservative":
        config.enable_conservative_mode()
    elif phase == "moderate":
        config.enable_moderate_mode()
    elif phase == "full":
        config.enable_full_mode()
    else:
        logger.error(f"❌ Unknown phase: {phase}")
        return

    logger.info(f"🔄 Convergence features enabled at phase: {phase}")


def emergency_disable_convergence():
    """Emergency disable all convergence features"""
    config = get_convergence_config()
    config.emergency_disable()


def get_convergence_status() -> Dict[str, Any]:
    """Get current convergence configuration status"""
    config = get_convergence_config()
    return config.get_status_summary()


# Example usage and testing functions
if __name__ == "__main__":
    # Test configuration system
    print("🧪 TESTING CONVERGENCE CONFIGURATION SYSTEM")
    print("=" * 60)

    config = get_convergence_config()

    # Test default state
    print("1. Default Configuration:")
    status = config.get_status_summary()
    print(f"   Enabled: {status['enabled']}")
    print(f"   Phase: {status['rollout_phase']}")
    print(f"   Safety: {status['safety_mode']}")

    # Test monitoring mode
    print("\n2. Enabling Monitoring Mode:")
    config.enable_monitoring_only()
    status = config.get_status_summary()
    print(f"   Enabled: {status['enabled']}")
    print(f"   Phase: {status['rollout_phase']}")

    # Test conservative mode
    print("\n3. Enabling Conservative Mode:")
    config.enable_conservative_mode()
    status = config.get_status_summary()
    print(f"   Enabled: {status['enabled']}")
    print(f"   Phase: {status['rollout_phase']}")
    print(f"   Features: {status['features_status']}")

    # Test emergency disable
    print("\n4. Emergency Disable:")
    config.emergency_disable()
    status = config.get_status_summary()
    print(f"   Enabled: {status['enabled']}")
    print(f"   Phase: {status['rollout_phase']}")

    print("\n✅ Configuration system test completed")
