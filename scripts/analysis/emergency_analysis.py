#!/usr/bin/env python3
"""
EMERGENCY ANALYSIS - 46% LOSS INVESTIGATION
==========================================

This script analyzes the catastrophic 46% loss to identify root causes.
Run this immediately after stopping the system.
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_portfolio_history():
    """Load portfolio history from CSV files"""
    portfolio_files = [
        "data/portfolios/portfolio_log.csv",
        "portfolio_state_live.json",
        "portfolio_state.json"
    ]

    history = []

    # Load CSV history
    csv_file = "data/portfolios/portfolio_log.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            history.extend(df.to_dict('records'))
            print(f"✅ Loaded {len(df)} records from portfolio CSV")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")

    # Load JSON states
    for json_file in ["portfolio_state_live.json", "portfolio_state.json"]:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    state = json.load(f)
                if 'portfolio' in state:
                    record = {
                        'timestamp': state.get('timestamp', datetime.utcnow().isoformat()),
                        'total_value': state['portfolio'].get('total', 0),
                        'btc_balance': state['portfolio'].get('BTCUSDT', {}).get('position', 0),
                        'eth_balance': state['portfolio'].get('ETHUSDT', {}).get('position', 0),
                        'usdt_balance': state['portfolio'].get('USDT', {}).get('free', 0),
                        'source': json_file
                    }
                    history.append(record)
                    print(f"✅ Loaded state from {json_file}")
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")

    return sorted(history, key=lambda x: x.get('timestamp', ''))

def analyze_portfolio_changes(history):
    """Analyze portfolio value changes over time"""
    if not history:
        print("❌ No portfolio history found")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(history)

    # Ensure numeric columns
    numeric_cols = ['total_value', 'btc_balance', 'eth_balance', 'usdt_balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate changes
    if len(df) > 1:
        df = df.sort_values('timestamp')
        df['value_change'] = df['total_value'].diff()
        df['value_change_pct'] = df['total_value'].pct_change() * 100

        # Find largest losses
        large_losses = df[df['value_change_pct'] < -5].copy()  # Losses > 5%
        if not large_losses.empty:
            print("\n🚨 LARGE LOSSES DETECTED:")
            print(large_losses[['timestamp', 'total_value', 'value_change', 'value_change_pct']].to_string())

    return df

def analyze_stop_loss_failures():
    """Analyze why stop-loss didn't protect capital"""
    print("\n🛡️ STOP-LOSS ANALYSIS:")

    # Check order manager configuration
    try:
        from l1_operational.order_manager import OrderManager
        print(f"✅ OrderManager MIN_ORDER_SIZE: ${OrderManager.MIN_ORDER_SIZE}")
    except Exception as e:
        print(f"❌ Error checking OrderManager: {e}")

    # Check if stop-loss orders were generated
    log_files = [
        f for f in os.listdir("logs") if f.endswith(".log")
    ] if os.path.exists("logs") else []

    stop_loss_count = 0
    for log_file in log_files[-3:]:  # Check last 3 log files
        try:
            with open(f"logs/{log_file}", 'r') as f:
                content = f.read()
                stop_loss_count += content.count("STOP-LOSS")
                if "STOP-LOSS" in content:
                    print(f"📄 Found stop-loss activity in {log_file}")
        except:
            pass

    if stop_loss_count == 0:
        print("❌ NO STOP-LOSS ORDERS FOUND IN LOGS")
        print("   This indicates stop-loss protection failed completely")
    else:
        print(f"✅ Found {stop_loss_count} stop-loss references in logs")

def analyze_signal_quality():
    """Analyze L2 signal quality and calibration"""
    print("\n🎯 L2 SIGNAL ANALYSIS:")

    # Check recent signals
    try:
        from l2_tactic.models import TacticalSignal
        print("✅ TacticalSignal class available")
    except Exception as e:
        print(f"❌ Error importing TacticalSignal: {e}")

    # Check FinRL model status
    model_paths = [
        "models/L2/deepsek.zip",
        "models/L2/finrl_model.zip"
    ]

    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Model found: {model_path}")
        else:
            print(f"❌ Model missing: {model_path}")

def analyze_market_timing():
    """Analyze operation timing vs market movements"""
    print("\n⏰ MARKET TIMING ANALYSIS:")

    # This would require market data analysis
    # For now, just check if we have recent market data
    market_data_files = [
        f for f in os.listdir("data") if f.startswith("market_data")
    ] if os.path.exists("data") else []

    if market_data_files:
        print(f"✅ Found {len(market_data_files)} market data files")
        for f in market_data_files[-3:]:  # Show last 3
            print(f"   - {f}")
    else:
        print("❌ No market data files found")

def generate_emergency_report():
    """Generate comprehensive emergency report"""
    print("\n" + "="*80)
    print("🚨 EMERGENCY ANALYSIS REPORT - 46% LOSS INVESTIGATION")
    print("="*80)

    # Load and analyze portfolio history
    history = load_portfolio_history()
    if history:
        df = analyze_portfolio_changes(history)

        if df is not None and len(df) > 0:
            # Calculate total loss
            initial_value = df['total_value'].iloc[0] if not df.empty else 3000
            final_value = df['total_value'].iloc[-1] if not df.empty else 0
            total_loss = ((final_value - initial_value) / initial_value) * 100

            print("\n💰 PORTFOLIO SUMMARY:")
            print(f"   Initial Value: ${initial_value:.2f}")
            print(f"   Final Value: ${final_value:.2f}")
            print(f"   Total Loss: {total_loss:.2f}%")
            if total_loss < -40:
                print("🚨 CONFIRMED: Catastrophic loss >40% detected")
            elif total_loss < -20:
                print("⚠️  SIGNIFICANT: Major loss >20% detected")
            else:
                print("ℹ️  MODERATE: Loss <20%")

    # Analyze stop-loss failures
    analyze_stop_loss_failures()

    # Analyze signal quality
    analyze_signal_quality()

    # Analyze market timing
    analyze_market_timing()

    print("\n" + "="*80)
    print("🔍 ROOT CAUSE HYPOTHESES:")
    print("="*80)

    hypotheses = [
        "1. Stop-loss orders not being generated or executed",
        "2. L2 FinRL signals giving consistently wrong directions",
        "3. Market timing issues - buying highs, selling lows",
        "4. Position sizing too aggressive despite limits",
        "5. L3 disabled but L2 still using stale/incorrect signals",
        "6. Order execution failures or slippage issues",
        "7. Portfolio state corruption or double-counting",
        "8. Market in extreme conditions system can't handle"
    ]

    for hypothesis in hypotheses:
        print(f"   • {hypothesis}")

    print("\n" + "="*80)
    print("🛠️  IMMEDIATE ACTION ITEMS:")
    print("="*80)

    actions = [
        "1. Keep system STOPPED until analysis complete",
        "2. Review all stop-loss logic in order_manager.py",
        "3. Backtest L2 signals on recent market data",
        "4. Check order execution logs for failures",
        "5. Verify portfolio state integrity",
        "6. Consider manual position management",
        "7. Implement circuit breakers for extreme losses"
    ]

    for action in actions:
        print(f"   • {action}")

    print("\n" + "="*80)
    print("📞 NEXT STEPS:")
    print("="*80)
    print("   1. Run: python emergency_analysis.py")
    print("   2. Review the detailed findings above")
    print("   3. Check logs in /logs directory")
    print("   4. Do NOT restart system until issues resolved")
    print("   5. Consider consulting with strategy experts")

if __name__ == "__main__":
    generate_emergency_report()
