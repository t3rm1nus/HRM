"""
Decision Maker - L3
Toma los outputs de todos los módulos de L3 (regime, sentiment, portfolio, risk, macro)
y genera las directrices estratégicas unificadas para L2.
ENHANCED: Now handles oversold/overbought setups within range regimes.
"""

import os
import json
from datetime import datetime
from .regime_classifier import clasificar_regimen_mejorado as clasificar_regimen
from .exposure_manager import gestionar_exposicion
from core.logging import logger

# Directorio de inferencias
INFER_DIR = "data/datos_inferencia"
OUTPUT_FILE = os.path.join(INFER_DIR, "strategic_decision.json")


def ensure_dir(directory: str):
    """Crea el directorio si no existe"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_inputs():
    """Carga todos los JSON disponibles en data/datos_inferencia"""
    results = {}
    for file in os.listdir(INFER_DIR):
        if file.endswith(".json") and file not in ["l3_output.json", "strategic_decision.json"]:
            path = os.path.join(INFER_DIR, file)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    results[file.replace(".json", "")] = data
            except Exception as e:
                print(f"⚠️ Error cargando {file}: {e}")
    return results


def make_decision(inputs: dict, portfolio_state: dict = None, market_data: dict = None, regime_decision: dict = None):
    """
    Combina todos los outputs de L3 en una decisión estratégica.
    ENHANCED: Now handles oversold/overbought setups within range regimes.
    
    PRIORITY HIERARCHY:
    1. Regime-specific decisions (including setups)
    2. Sentiment adjustments
    3. Risk management overlays
    """
    # ========================================================================================
    # PRIORITY 1: REGIME-SPECIFIC DECISIONS WITH SETUP DETECTION
    # ========================================================================================
    if regime_decision and isinstance(regime_decision, dict):
        regime = regime_decision.get('regime', 'neutral')
        subtype = regime_decision.get('subtype', None)
        regime_signal = regime_decision.get('signal', 'hold')
        regime_confidence = regime_decision.get('confidence', 0.5)
        setup_type = regime_decision.get('setup_type', None)
        allow_l2_signal = regime_decision.get('allow_l2_signal', False)

        # Get current prices for allocation calculations
        current_btc_price = None
        current_eth_price = None

        if market_data:
            if "BTCUSDT" in market_data:
                btc_data = market_data["BTCUSDT"]
                if isinstance(btc_data, dict) and 'close' in btc_data:
                    current_btc_price = float(btc_data['close'])
                elif hasattr(btc_data, 'iloc') and len(btc_data) > 0:
                    try:
                        current_btc_price = float(btc_data['close'].iloc[-1])
                    except:
                        pass

            if "ETHUSDT" in market_data:
                eth_data = market_data["ETHUSDT"]
                if isinstance(eth_data, dict) and 'close' in eth_data:
                    current_eth_price = float(eth_data['close'])
                elif hasattr(eth_data, 'iloc') and len(eth_data) > 0:
                    try:
                        current_eth_price = float(eth_data['close'].iloc[-1])
                    except:
                        pass

        # ========================================================================================
        # NEW: SETUP-BASED ALLOCATIONS (Override standard regime logic)
        # ========================================================================================
        if setup_type == 'oversold':
            # OVERSOLD SETUP: Small buy allocation for mean reversion
            portfolio = {
                'BTCUSDT': 0.15,  # 15% allocation
                'ETHUSDT': 0.10,  # 10% allocation
                'USDT': 0.75      # Keep 75% liquid
            }
            risk_appetite = 'moderate'
            logger.info(f"🎯 OVERSOLD SETUP: Allowing limited L2 BUY signals with {regime_confidence:.2f} confidence")
            logger.info(f"📊 Target allocation: BTC 15%, ETH 10%, USDT 75%")
            
        elif setup_type == 'overbought':
            # OVERBOUGHT SETUP: Small short allocation or exit longs
            portfolio = {
                'BTCUSDT': 0.05,  # Reduce to 5%
                'ETHUSDT': 0.05,  # Reduce to 5%
                'USDT': 0.90      # Move to 90% cash
            }
            risk_appetite = 'low'
            logger.info(f"🎯 OVERBOUGHT SETUP: Allowing limited L2 SELL signals with {regime_confidence:.2f} confidence")
            logger.info(f"📊 Target allocation: BTC 5%, ETH 5%, USDT 90%")
            
        # ========================================================================================
        # STANDARD REGIME ALLOCATIONS (when no setup detected)
        # ========================================================================================
        elif regime_signal == 'buy':
            if regime == 'bull':
                # Aggressive allocations in bull markets
                portfolio = {
                    'BTCUSDT': regime_confidence * 0.40,
                    'ETHUSDT': regime_confidence * 0.30,
                    'USDT': max(0.10, 1.0 - regime_confidence * 0.70)
                }
            elif regime == 'bear':
                # Conservative allocations in bear markets
                portfolio = {
                    'BTCUSDT': regime_confidence * 0.02,
                    'ETHUSDT': regime_confidence * 0.02,
                    'USDT': max(0.10, 1.0 - regime_confidence * 0.04)
                }
            else:  # range, neutral
                # Balanced allocations
                portfolio = {
                    'BTCUSDT': regime_confidence * 0.20,
                    'ETHUSDT': regime_confidence * 0.15,
                    'USDT': max(0.10, 1.0 - regime_confidence * 0.35)
                }
        else:
            # Default HOLD allocations
            portfolio = {
                'BTCUSDT': 0.20,
                'ETHUSDT': 0.15,
                'USDT': 0.65
            }

        logger.info(f"🎯 REGIME PRIORITY: {regime.upper()} regime (subtype: {subtype}) with {regime_confidence:.2f} confidence")
        if setup_type:
            logger.info(f"🎯 SETUP DETECTED: {setup_type} - Mean reversion strategy active")

        # Override risk appetite based on regime and setup
        if setup_type:
            risk_appetite = 'moderate'  # Always moderate for setups
        elif regime == 'bull':
            risk_appetite = 'high'
        elif regime == 'bear':
            risk_appetite = 'low'
        else:
            risk_appetite = 'moderate'

    else:
        # Fallback when no regime decision provided
        regime = clasificar_regimen(market_data) if market_data else inputs.get("regime_detection", {}).get("predicted_regime", "neutral")
        portfolio = inputs.get("portfolio", {}).get("weights", {})
        subtype = None
        setup_type = None
        allow_l2_signal = False

    # Get remaining inputs
    sentiment = inputs.get("sentiment", {}).get("sentiment_score", 0.0)
    if 'risk_appetite' not in locals():
        risk_appetite = inputs.get("risk", {}).get("risk_appetite", "moderate")
    macro = inputs.get("macro", {})

    # ========================================================================================
    # EXPOSURE MANAGEMENT
    # ========================================================================================
    exposure_decisions = {}
    if portfolio_state and market_data:
        universo = ["BTCUSDT", "ETHUSDT"]
        exposure_decisions = gestionar_exposicion(universo, portfolio_state, market_data, regime)
        logger.info("📊 Decisiones de exposición calculadas exitosamente")
    else:
        logger.warning("⚠️ Datos insuficientes para gestión de exposición - usando configuración por defecto")

    # ========================================================================================
    # CALIBRATED EXPOSURE GUIDELINES
    # ========================================================================================
    max_single_exposure = 0.8 if risk_appetite == "high" else 0.6
    if regime == "bear":
        max_single_exposure = 0.5
    elif regime == "bull":
        max_single_exposure = 0.9
    
    # Adjust for setups
    if setup_type == 'oversold':
        max_single_exposure = 0.20  # Limited exposure on oversold setups
    elif setup_type == 'overbought':
        max_single_exposure = 0.10  # Minimal exposure on overbought setups

    # ========================================================================================
    # LOSS PREVENTION FILTERS WITH SETUP OVERRIDES
    # ========================================================================================
    loss_prevention_filters = {
        "max_loss_per_trade_pct": 0.035,
        "require_strong_signal": False,
        "avoid_weak_sentiment": sentiment < -0.5,
        "bear_market_restriction": False,
        "high_volatility_block": False,
        "preserve_high_conf_l2": True,
        "allow_setup_trades": setup_type is not None,  # NEW: Allow trades on setups
        "setup_type": setup_type  # NEW: Pass setup type to L2
    }

    # Check volatility
    volatility_data = inputs.get("volatility", {})
    if volatility_data:
        btc_vol = volatility_data.get("btc_volatility", 0.03)
        eth_vol = volatility_data.get("eth_volatility", 0.04)
        avg_vol = (btc_vol + eth_vol) / 2
        if avg_vol > 0.05:
            loss_prevention_filters["high_volatility_block"] = True
            loss_prevention_filters["max_loss_per_trade_pct"] = 0.02

    # ========================================================================================
    # WINNING TRADE RULES WITH SETUP ADJUSTMENTS
    # ========================================================================================
    winning_trade_rules = {
        "allow_profit_running": True,
        "trailing_stop_activation": 0.005,
        "take_profit_levels": [0.03, 0.08, 0.15, 0.25],
        "scale_out_profits": True,
        "hold_winners_longer": regime in ["bull", "range"],
        "momentum_boost": True,
        "early_exit_weak_signals": False,
        "profit_lock_in": 0.02,
    }
    
    # Adjust for setups (tighter targets for mean reversion)
    if setup_type:
        winning_trade_rules["take_profit_levels"] = [0.015, 0.025, 0.04]  # Tighter targets
        winning_trade_rules["trailing_stop_activation"] = 0.008
        winning_trade_rules["profit_lock_in"] = 0.012

    # ========================================================================================
    # FINAL DECISION STRUCTURE
    # ========================================================================================
    decision = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_regime": regime,
        "regime_subtype": subtype,
        "setup_detected": setup_type is not None,
        "setup_type": setup_type,
        "allow_l2_signals": allow_l2_signal or (setup_type is not None),
        "sentiment_score": sentiment,
        "asset_allocation": portfolio,
        "risk_appetite": risk_appetite,
        "macro_context": macro,
        "exposure_decisions": exposure_decisions,
        "loss_prevention_filters": loss_prevention_filters,
        "winning_trade_rules": winning_trade_rules,
        "strategic_guidelines": {
            "rebalance_frequency": "daily" if regime == "volatile" else "weekly",
            "max_single_asset_exposure": max_single_exposure,
            "volatility_target": 0.25 if risk_appetite == "high" else 0.15,
            "liquidity_requirement": "high" if risk_appetite != "high" or regime == "bear" else "medium",
            "btc_max_exposure": 0.3 if regime == "bear" else 0.6,
            "usdt_min_liquidity": 0.10,
            "max_loss_per_trade_pct": loss_prevention_filters["max_loss_per_trade_pct"],
            "require_stop_loss": True,
            "profit_taking_strategy": "scaled" if winning_trade_rules["scale_out_profits"] else "single_target",
            "setup_trading_enabled": setup_type is not None
        }
    }
    
    # Log key decision points
    if setup_type:
        logger.info(f"✅ SETUP STRATEGY ACTIVE: {setup_type} setup allows controlled L2 signals")
        logger.info(f"📊 Setup allocation: BTC {portfolio.get('BTCUSDT', 0):.1%}, ETH {portfolio.get('ETHUSDT', 0):.1%}, USDT {portfolio.get('USDT', 0):.1%}")
    
    return decision


def save_decision(data: dict, output_path: str):
    """Guarda la decisión estratégica en JSON"""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Decisión estratégica guardada en {output_path}")


if __name__ == "__main__":
    print("🔄 Ejecutando Decision Maker...")
    ensure_dir(INFER_DIR)

    inputs = load_inputs()
    decision = make_decision(inputs)
    save_decision(decision, OUTPUT_FILE)

    print("📊 Resumen Decision Maker:")
    print(json.dumps(decision, indent=4))