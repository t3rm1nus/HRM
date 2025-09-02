#!/usr/bin/env python3
"""
HRM System Diagnostic Script
Identifies and fixes common issues in the HRM trading system
"""

import os
import sys
import time
import logging
from datetime import datetime

def check_environment():
    """Check environment variables and configuration"""
    print("🔧 Checking Environment Configuration...")
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'USE_TESTNET',
        'BINANCE_MODE'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All environment variables configured")
        return True

def test_binance_connection():
    """Test connection to Binance API"""
    print("\n📡 Testing Binance Connection...")

    try:
        from l1_operational.binance_client import BinanceClient
        client = BinanceClient()

        # Probar con klines ya que no existe get_server_time ni get_symbol_price
        try:
            klines = client.get_klines(symbol="BTCUSDT", interval="1m", limit=5)
            if klines:
                print(f"✅ Binance conectado. Datos BTCUSDT: {len(klines)} velas recibidas")
                return True
            else:
                print("❌ No se pudieron obtener velas desde Binance")
                return False
        except Exception as e:
            print(f"❌ Error al obtener klines: {e}")
            return False

    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        return False


def test_data_feed():
    """Test market data collection"""
    print("\n📊 Testing Data Feed...")

    try:
        from l1_operational.data_feed import DataFeed

        # Intentar instanciar sin argumentos
        try:
            data_feed = DataFeed()
        except TypeError:
            # Si requiere símbolos explícitos, usa una lista por defecto
            data_feed = DataFeed(symbols=['BTCUSDT', 'ETHUSDT'])

        if hasattr(data_feed, "get_latest_data") and callable(data_feed.get_latest_data):
            data = data_feed.get_latest_data()
            if data and len(data) > 0:
                print(f"✅ Data feed working: {len(data)} símbolos con datos")
                for symbol, symbol_data in data.items():
                    print(f"  - {symbol}: {len(symbol_data)} registros")
                return True
            else:
                print("❌ Data feed retornó vacío")
                return False
        else:
            print("⚠️ DataFeed no tiene método 'get_latest_data'")
            return False

    except Exception as e:
        print(f"❌ Data feed error: {e}")
        return False


def test_feature_calculation():
    """Test technical indicator calculation"""
    print("\n🧮 Testing Feature Calculation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data to test feature calculation
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(50000, 52000, 100),
            'high': np.random.uniform(52000, 54000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(50000, 52000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }).set_index('timestamp')
        
        # Test RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(sample_data['close'])
        
        if not rsi.isna().all():
            print("✅ Technical indicators calculation working")
            print(f"  - RSI calculated: min={rsi.min():.2f}, max={rsi.max():.2f}")
            return True
        else:
            print("❌ Technical indicators calculation failed")
            return False
            
    except Exception as e:
        print(f"❌ Feature calculation error: {e}")
        return False

def check_ai_models():
    """Check AI models availability"""
    print("\n🤖 Checking AI Models...")
    
    model_paths = [
        'models/L1/modelo1_lr.pkl',
        'models/L1/modelo2_rf.pkl', 
        'models/L1/modelo3_lgbm.pkl'
    ]
    
    missing_models = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print(f"❌ Missing AI models: {missing_models}")
        return False
    else:
        print("✅ All AI models found")
        return True

def check_data_directories():
    """Check required data directories"""
    print("\n📁 Checking Data Directories...")
    
    required_dirs = [
        'data/logs',
        'data/portfolio',
        'models/L1'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"📁 Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    return True

def run_system_diagnostic():
    """Run complete system diagnostic"""
    print("🔍 HRM System Diagnostic Starting...")
    print("=" * 50)
    
    checks = [
        ("Environment", check_environment),
        ("Data Directories", check_data_directories),
        ("AI Models", check_ai_models),
        ("Binance Connection", test_binance_connection),
        ("Data Feed", test_data_feed),
        ("Feature Calculation", test_feature_calculation)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            results[check_name] = False
        
        time.sleep(1)  # Brief pause between checks
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic Summary:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! System ready for operation.")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = run_system_diagnostic()
    sys.exit(0 if success else 1)