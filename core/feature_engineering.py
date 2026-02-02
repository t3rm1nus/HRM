# core/feature_engineering.py
import pandas as pd
from core.logging import logger
import json
from typing import Dict, Optional, List, Any


def integrate_features_with_l2(technical_indicators: dict, l3_context: dict) -> dict:
    """
    Integrate technical indicators with L3 strategic context.
    
    Args:
        technical_indicators (dict): Dict of DataFrames with technical indicators per symbol
        l3_context (dict): L3 strategic context dict from state['estrategia']
    
    Returns:
        dict: Technical indicators enriched with L3 features
    
    Raises:
        ValueError: If inputs are not dictionaries or have invalid format
    """
    # Validación de inputs
    if not isinstance(technical_indicators, dict):
        raise ValueError("technical_indicators debe ser un diccionario")
    if not isinstance(l3_context, dict):
        raise ValueError("l3_context debe ser un diccionario")

    # Crear copia para no modificar el original
    enriched_indicators = {}
    
    try:
        # Procesar cada símbolo
        for symbol, df in technical_indicators.items():
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Saltando {symbol}: no es un DataFrame")
                continue
            if df.empty:
                logger.warning(f"Saltando {symbol}: DataFrame vacío")
                continue
            if 'close' not in df.columns:
                logger.warning(f"Saltando {symbol}: falta columna 'close'")
                continue
                
            try:
                # Copiar el DataFrame original
                enriched_df = df.copy()
                
                # Añadir features de L3 con validación
                regime = str(l3_context.get('regime', 'neutral')).lower()
                regime_value = {
                    'bull': 1.0,
                    'neutral': 0.5,
                    'bear': 0.0
                }.get(regime, 0.5)
                enriched_df['l3_regime'] = regime_value
                
                try:
                    risk_appetite = float(l3_context.get('risk_appetite', 0.5))
                    risk_appetite = max(0.0, min(1.0, risk_appetite))  # Clamp entre 0 y 1
                except (ValueError, TypeError):
                    risk_appetite = 0.5
                enriched_df['l3_risk_appetite'] = risk_appetite
                
                # Validar y procesar asset allocation
                asset_allocation = l3_context.get('asset_allocation', {})
                if not isinstance(asset_allocation, dict):
                    asset_allocation = {}
                    
                for asset in ['BTC', 'ETH', 'CASH']:
                    try:
                        value = float(asset_allocation.get(asset, 0.0))
                        value = max(0.0, min(1.0, value))  # Clamp entre 0 y 1
                    except (ValueError, TypeError):
                        value = 0.0 if asset != 'CASH' else 1.0
                    enriched_df[f'l3_alloc_{asset}'] = value
                
                # Calcular ratios entre pares con validación
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    other_symbol = 'ETHUSDT' if symbol == 'BTCUSDT' else 'BTCUSDT'
                    if other_symbol in technical_indicators:
                        other_df = technical_indicators[other_symbol]
                        if not other_df.empty and 'close' in other_df.columns:
                            try:
                                # Asegurar que los close son numéricos
                                symbol_close = pd.to_numeric(enriched_df['close'], errors='coerce')
                                other_close = pd.to_numeric(other_df['close'], errors='coerce')
                                
                                if symbol == 'BTCUSDT':
                                    ratio = other_close / symbol_close
                                else:  # ETHUSDT
                                    ratio = symbol_close / other_close
                                    
                                enriched_df['eth_btc_ratio'] = ratio.fillna(method='ffill').fillna(1.0)
                                
                                # Correlación con ventana mínima de datos válidos
                                valid_mask = ~(symbol_close.isna() | other_close.isna())
                                if valid_mask.sum() >= 5:  # Mínimo 5 puntos válidos
                                    corr = symbol_close.rolling(10, min_periods=5).corr(other_close)
                                    enriched_df['rolling_corr_10'] = corr.fillna(method='ffill')
                                else:
                                    enriched_df['rolling_corr_10'] = pd.NA
                                    
                            except Exception as calc_error:
                                logger.error(f"Error calculando ratios para {symbol}: {calc_error}")
                                enriched_df['eth_btc_ratio'] = 1.0
                                enriched_df['rolling_corr_10'] = pd.NA
                
                # Guardar el DataFrame enriquecido
                enriched_indicators[symbol] = enriched_df
                
                # Debug info detallado
                l3_cols = [col for col in enriched_df.columns if col.startswith('l3_')]
                logger.debug(f"{symbol} enriched with {len(l3_cols)} L3 features: {l3_cols}")
                
            except Exception as symbol_error:
                logger.error(f"Error procesando {symbol}: {symbol_error}")
                enriched_indicators[symbol] = df  # Mantener datos originales en caso de error
            
        return enriched_indicators
        
    except Exception as e:
        logger.error(f"❌ Error general enriqueciendo features con L3: {e}", exc_info=True)
        return technical_indicators

# ----------------------
# Debug L2 Features
# ----------------------
def debug_l2_features(features: dict):
    """
    Debug de features L2 enfocado en datos de L3.
    
    Args:
        features (dict): Dict de DataFrames con indicadores por símbolo
    """
    logger.info("=== L2 Features L3 Integration Debug ===")
    l3_columns = ['l3_regime', 'l3_risk_appetite', 'l3_alloc_BTC', 'l3_alloc_ETH', 'l3_alloc_CASH']
    cross_columns = ['eth_btc_ratio', 'rolling_corr_10']
    
    for symbol, df in features.items():
        if df is None or df.empty:
            logger.warning(f"{symbol}: DataFrame vacío")
            continue
            
        # Debug L3 features
        if any(col in df.columns for col in l3_columns):
            last_row = df.iloc[-1]
            l3_values = {col: last_row.get(col, 'N/A') for col in l3_columns if col in df.columns}
            logger.info(f"{symbol} L3 features: {l3_values}")
        else:
            logger.warning(f"{symbol}: No L3 features found")
            
        # Debug cross features
        if any(col in df.columns for col in cross_columns):
            last_row = df.iloc[-1]
            cross_values = {col: last_row.get(col, 'N/A') for col in cross_columns if col in df.columns}
            logger.info(f"{symbol} Cross features: {cross_values}")
        else:
            logger.warning(f"{symbol}: No cross features found")
            
        # Shape y memoria
        logger.info(f"{symbol} shape: {df.shape}, memoria: {df.memory_usage().sum() / 1024:.2f} KB")