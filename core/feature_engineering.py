# core/feature_engineering.py
import pandas as pd
from core.logging import logger
import json


def integrate_features_with_l2(df_symbol, df_other, l3_path):
    """
    Integrate technical indicators, cross-asset features, and L3 strategic context.
    
    Args:
        df_symbol (pd.DataFrame): Indicators for the primary symbol (e.g., BTCUSDT).
        df_other (pd.DataFrame): Indicators for other symbols (e.g., ETHUSDT).
        l3_path (str): Path to L3 output JSON.
    
    Returns:
        pd.DataFrame: Integrated features or empty DataFrame with expected columns if failed.
    """
    expected_columns = [
        'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'delta_close',
        'eth_btc_ratio', 'rolling_corr_10', 'l3_regime', 'l3_risk_appetite',
        'l3_alloc_BTC', 'l3_alloc_ETH', 'l3_alloc_CASH'
    ]
    
    if df_symbol is None or not isinstance(df_symbol, pd.DataFrame) or df_symbol.empty:
        logger.warning(f"⚠️ Invalid or empty df_symbol, returning empty DataFrame")
        return pd.DataFrame(columns=expected_columns)
    
    try:
        logger.debug(f"Processing features with shape: {df_symbol.shape}, columns: {list(df_symbol.columns)}")
        
        # Updated to match actual column names from calculate_technical_indicators
        required_cols = ['close', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']
        missing_cols = [col for col in required_cols if col not in df_symbol.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame(columns=expected_columns)
        
        # Copy required columns and rename to expected output format
        features = df_symbol[required_cols].copy()
        features = features.rename(columns={
            'RSI_14': 'rsi',
            'MACD': 'macd',
            'BB_upper': 'bollinger_upper',
            'BB_lower': 'bollinger_lower'
        })
        
        # Handle missing values
        features = features.fillna(0.0)
        
        # Add cross-asset features
        if df_other is not None and not df_other.empty and 'close' in df_other.columns:
            features['eth_btc_ratio'] = df_symbol['close'] / df_other['close']
            features['rolling_corr_10'] = df_symbol['close'].rolling(window=10).corr(df_other['close'])
            features['rolling_corr_10'] = features['rolling_corr_10'].fillna(0.0)
        else:
            logger.warning("⚠️ df_other is empty or missing 'close' column")
            features['eth_btc_ratio'] = 0.0
            features['rolling_corr_10'] = 0.0
        
        # Load L3 strategic context
        try:
            with open(l3_path, 'r') as f:
                l3_data = json.load(f)
            features['l3_regime'] = l3_data.get('regime', 'neutral')
            features['l3_risk_appetite'] = l3_data.get('risk_appetite', 'moderate')
            features['l3_alloc_BTC'] = l3_data.get('asset_allocation', {}).get('BTC', 0.5)
            features['l3_alloc_ETH'] = l3_data.get('asset_allocation', {}).get('ETH', 0.5)
            features['l3_alloc_CASH'] = l3_data.get('asset_allocation', {}).get('CASH', 0.0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load L3 data from {l3_path}: {e}")
            features['l3_regime'] = 'neutral'
            features['l3_risk_appetite'] = 'moderate'
            features['l3_alloc_BTC'] = 0.5
            features['l3_alloc_ETH'] = 0.5
            features['l3_alloc_CASH'] = 0.0
        
        # Add delta_close
        features['delta_close'] = df_symbol['close'].pct_change().fillna(0.0)
        
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in features.columns:
                features[col] = 0.0
        
        logger.debug(f"Features integrated successfully, shape: {features.shape}")

        # Añadir columna market_data con estructura esperada por FinRL
        if not features.empty and isinstance(df_symbol, pd.DataFrame) and not df_symbol.empty:
            last_row = df_symbol.iloc[-1].to_dict()
            # Separar OHLCV y los indicadores relevantes
            ohlcv_keys = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored']
            indicator_keys = ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'sma_20', 'ema_12', 'sma_10', 'ema_10']
            ohlcv = {k: last_row[k] for k in ohlcv_keys if k in last_row}
            indicators = {k: last_row[k] for k in indicator_keys if k in last_row}
            # Para compatibilidad, también incluir variantes de nombres
            if 'RSI_14' in last_row and 'rsi' not in indicators:
                indicators['rsi'] = last_row['RSI_14']
            if 'MACD' in last_row and 'macd' not in indicators:
                indicators['macd'] = last_row['MACD']
            if 'MACD_signal' in last_row and 'macd_signal' not in indicators:
                indicators['macd_signal'] = last_row['MACD_signal']
            if 'BB_upper' in last_row and 'bollinger_upper' not in indicators:
                indicators['bollinger_upper'] = last_row['BB_upper']
            if 'BB_lower' in last_row and 'bollinger_lower' not in indicators:
                indicators['bollinger_lower'] = last_row['BB_lower']
            if 'SMA_20' in last_row and 'sma_20' not in indicators:
                indicators['sma_20'] = last_row['SMA_20']
            if 'EMA_12' in last_row and 'ema_12' not in indicators:
                indicators['ema_12'] = last_row['EMA_12']
            if 'SMA_10' in last_row and 'sma_10' not in indicators:
                indicators['sma_10'] = last_row['SMA_10']
            if 'EMA_10' in last_row and 'ema_10' not in indicators:
                indicators['ema_10'] = last_row['EMA_10']
            # Construir market_data estructurado
            market_data_struct = {'ohlcv': ohlcv, 'indicators': indicators}
            features['market_data'] = [market_data_struct] * len(features)
        else:
            features['market_data'] = [{}] * len(features)

        return features
    
    except Exception as e:
        logger.error(f"❌ Error integrating features: {e}", exc_info=True)
        return pd.DataFrame(columns=expected_columns)
# ----------------------
# Debug L2 Features
# ----------------------
def debug_l2_features(features: dict, n: int = 5):
    """
    Debug de features L2.
    features: dict[symbol -> pd.DataFrame]
    """
    logger.info("=== L2 Features Preview ===")
    for symbol, df in features.items():
        if df is None or df.empty:
            logger.warning(f"{symbol}: DataFrame vacío")
            continue
        logger.info(f"{symbol} head:\n{df.head(n)}")
        logger.info(f"{symbol} dtypes:\n{df.dtypes.value_counts()}")
        logger.info(f"{symbol} memoria: {df.memory_usage().sum() / 1024:.2f} KB")