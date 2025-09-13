"""
Functions to prepare features for regime detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las features necesarias para regime detection, incluyendo retornos
    en múltiples timeframes y volatilidades.
    
    Args:
        df: DataFrame con columnas OHLCV
        
    Returns:
        DataFrame con las features para regime detection
    """
    from core import logging as log
    
    # Verificar y convertir datos a numérico
    numeric_df = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        try:
            numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            log.error(f"Error convirtiendo {col} a numérico: {e}")
            numeric_df[col] = 0.0
    
    # Inicializar DataFrame de features
    features = pd.DataFrame(index=df.index)
    
    # OHLCV base (ya convertido a numérico)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        features[col] = numeric_df[col].fillna(method='ffill')  # Forward fill para NaN
    
    try:
        returns = features['close'].pct_change()
        features['return'] = returns.fillna(0)
        features['log_return'] = np.log1p(returns).fillna(0)
        
        # Volatility features (exactly matching training)
        for w in [5, 15, 30, 60, 120]:
            # Return based
            features[f'volatility_{w}'] = returns.rolling(w, min_periods=1).std().fillna(0)
            features[f'return_{w}'] = returns.rolling(w, min_periods=1).mean().fillna(0)
            
        # RSI standard 14-period
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)  # Evitar división por cero
        features['rsi'] = (100 - (100 / (1 + rs))).fillna(50)  # 50 es neutral
        
        # Bollinger Bands (standard 20-period)
        ma20 = features['close'].rolling(window=20, min_periods=1).mean()
        std20 = features['close'].rolling(window=20, min_periods=1).std()
        features['boll_middle'] = ma20.fillna(method='ffill')
        features['boll_upper'] = (ma20 + (std20 * 2)).fillna(method='ffill')
        features['boll_lower'] = (ma20 - (std20 * 2)).fillna(method='ffill')
        
        # MACD (standard 12,26,9)
        ema12 = features['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = features['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        features['macd'] = (ema12 - ema26).fillna(0)
        features['macdsig'] = features['macd'].ewm(span=9, adjust=False, min_periods=1).mean().fillna(0)
        features['macdhist'] = (features['macd'] - features['macdsig']).fillna(0)
        
    except Exception as e:
        log.error(f"Error calculando features técnicas: {e}")
        # Asegurar que todas las columnas existan aunque haya error
        for col in features.columns:
            if features[col].isna().any():
                features[col] = features[col].fillna(0)
    
    return features
    
    return features

def validate_regime_features(features: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Valida que todas las features requeridas estén presentes y
    rellena valores faltantes si es necesario.
    
    Args:
        features: DataFrame con las features calculadas
        required_features: Lista de features requeridas por el modelo
        
    Returns:
        DataFrame con todas las features requeridas, rellenando con 0.0 las faltantes
    """
    from core import logging as log
    
    # Verificar features faltantes
    missing_features = []
    zero_features = []
    nan_features = []
    extra_features = []
    
    # Identificar features extra
    for col in features.columns:
        if col not in required_features:
            extra_features.append(col)
            
    # Eliminar features extra para evitar confusión
    if extra_features:
        log.info(f"Eliminando {len(extra_features)} features no requeridas por el modelo")
        features = features.drop(columns=extra_features)
    
    for feature in required_features:
        if feature not in features.columns:
            missing_features.append(feature)
            features[feature] = 0.0
        else:
            if features[feature].isna().all():
                nan_features.append(feature)
                features[feature] = features[feature].fillna(0.0)
            elif (features[feature] == 0).all():
                zero_features.append(feature)
                
    # Agrupar features por tipo para mejor diagnóstico
    def _group_features(feat_list):
        groups = {
            'OHLCV': [], 'returns': [], 'technical': [], 'volatility': [], 'other': []
        }
        for f in feat_list:
            if f in ['open', 'high', 'low', 'close', 'volume']:
                groups['OHLCV'].append(f)
            elif 'return' in f:
                groups['returns'].append(f)
            elif any(t in f for t in ['macd', 'boll', 'rsi']):
                groups['technical'].append(f)
            elif 'vol' in f:
                groups['volatility'].append(f)
            else:
                groups['other'].append(f)
        return {k: v for k, v in groups.items() if v}
    
    # Log detallado del estado de las features
    if missing_features:
        grouped = _group_features(missing_features)
        log.warning("Features faltantes por tipo:")
        for group, feats in grouped.items():
            if feats:
                log.warning(f"- {group}: {', '.join(feats)}")
                
    # Log de estadísticas
    feature_stats = {
        'provided': len(features.columns),
        'required': len(required_features),
        'missing': len(missing_features),
        'zeros': len(zero_features),
        'nans': len(nan_features)
    }
    log.info(f"Estadísticas de features:")
    log.info(f"- Features proporcionadas: {feature_stats['provided']}")
    log.info(f"- Features requeridas: {feature_stats['required']}")
    if feature_stats['missing'] > 0:
        log.warning(f"- Features faltantes: {feature_stats['missing']}")
    if feature_stats['zeros'] > 0:
        log.warning(f"- Features en cero: {feature_stats['zeros']}")
    if feature_stats['nans'] > 0:
        log.warning(f"- Features con NaN: {feature_stats['nans']}")
                
    if zero_features:
        grouped = _group_features(zero_features)
        log.warning("Features en cero por tipo:")
        for group, feats in grouped.items():
            if feats:
                log.warning(f"- {group}: {', '.join(feats)}")
                
    if nan_features:
        grouped = _group_features(nan_features)
        log.warning("Features con NaN por tipo:")
        for group, feats in grouped.items():
            if feats:
                log.warning(f"- {group}: {', '.join(feats)}")
    
    # Asegurar que tenemos todas las features requeridas
    features = features[required_features].copy()
    
    # Verificación final de calidad
    na_count = features.isna().sum()
    if na_count.any():
        log.warning(f"⚠️ Hay {na_count.sum()} valores NaN después de la validación")
        features = features.fillna(0.0)
    
    return features
