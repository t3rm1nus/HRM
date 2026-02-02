"""
Validación de Datos de Mercado del Sistema HRM

Este módulo maneja la validación y extracción segura de datos de mercado,
incluyendo validación de estructura, tipos de datos y calidad de información.
"""

from typing import Dict, Any, List, Union
import pandas as pd
from datetime import datetime

from core.logging import logger


def validate_market_data(market_data: Dict) -> bool:
    """
    Valida la estructura de datos de mercado antes del despliegue.
    
    Args:
        market_data: Datos de mercado a validar
        
    Returns:
        bool: True si los datos son válidos, False en caso contrario
    """
    required_symbols = ['BTCUSDT', 'ETHUSDT']

    for symbol in required_symbols:
        if symbol not in market_data:
            logger.error(f"❌ Falta {symbol} en market_data")
            return False

        df = market_data[symbol]
        if isinstance(df, pd.DataFrame):
            if df.empty or 'close' not in df.columns:
                logger.error(f"❌ DataFrame inválido para {symbol}")
                return False
            price = df['close'].iloc[-1]
        elif isinstance(df, dict):
            if 'close' not in df:
                logger.error(f"❌ Falta campo 'close' en {symbol}")
                return False
            price = df['close']
        else:
            logger.error(f"❌ Tipo de datos no soportado para {symbol}: {type(df)}")
            return False

        if pd.isna(price) or price <= 0:
            logger.error(f"❌ Precio inválido para {symbol}: {price}")
            return False

    return True


def _extract_current_price_safely(symbol: str, market_data: Dict) -> float:
    """
    Extrae el precio actual de forma segura desde datos de mercado para validación y despliegue.
    
    Args:
        symbol: Símbolo de trading (ej. 'BTCUSDT')
        market_data: Diccionario con datos de mercado
        
    Returns:
        Precio actual como float, o 0.0 si la extracción falla
    """
    try:
        if not market_data or symbol not in market_data:
            return 0.0

        data = market_data[symbol]

        if isinstance(data, dict):
            if 'close' in data:
                return float(data.get('close', 0.0))
            elif 'price' in data:
                return float(data.get('price', 0.0))
        elif isinstance(data, pd.DataFrame):
            if 'close' in data.columns and len(data) > 0:
                return float(data['close'].iloc[-1])
        elif isinstance(data, pd.Series) and len(data) > 0:
            return float(data.iloc[-1])

        return 0.0
    except Exception as e:
        logger.error(f"❌ Error extrayendo precio para {symbol}: {e}")
        return 0.0


def validate_market_data_structure(market_data: Dict) -> tuple[bool, str]:
    """
    Valida la estructura completa de datos de mercado.
    
    Args:
        market_data: Datos de mercado a validar
        
    Returns:
        tuple[bool, str]: (es_válido, mensaje_de_validación)
    """
    if not market_data or not isinstance(market_data, dict):
        return False, "Datos de mercado vacíos o no es un diccionario"
    
    if not market_data:
        return False, "Datos de mercado están vacíos"
    
    # Validar símbolos requeridos
    required_symbols = ['BTCUSDT', 'ETHUSDT']
    missing_symbols = [sym for sym in required_symbols if sym not in market_data]
    
    if missing_symbols:
        return False, f"Símbolos requeridos faltantes: {missing_symbols}"
    
    # Validar estructura de cada símbolo
    for symbol, data in market_data.items():
        if not isinstance(data, (dict, pd.DataFrame)):
            return False, f"Tipo de datos inválido para {symbol}: {type(data)}"
        
        if isinstance(data, dict):
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return False, f"Campos requeridos faltantes en {symbol}: {missing_fields}"
            
            # Validar que los valores sean numéricos
            for field in required_fields:
                if not isinstance(data[field], (int, float)):
                    return False, f"Valor no numérico en {symbol}.{field}: {type(data[field])}"
        
        elif isinstance(data, pd.DataFrame):
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Columnas requeridas faltantes en {symbol}: {missing_columns}"
            
            if data.empty:
                return False, f"DataFrame vacío para {symbol}"
    
    return True, "Datos de mercado válidos"


def validate_and_fix_market_data(state: Dict, config: Dict) -> tuple[Dict, str]:
    """
    Valida y repara datos de mercado en el estado del sistema.
    
    Args:
        state: Estado del sistema
        config: Configuración del sistema
        
    Returns:
        tuple[Dict, str]: (datos_válidos, mensaje_de_validación)
    """
    market_data = state.get("market_data", {})
    
    is_valid, validation_msg = validate_market_data_structure(market_data)
    
    if is_valid:
        return market_data, validation_msg
    
    logger.warning(f"⚠️ Datos de mercado inválidos: {validation_msg}")
    
    # Intentar reparar datos
    fixed_data = {}
    required_symbols = config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    
    for symbol in required_symbols:
        if symbol in market_data:
            data = market_data[symbol]
            
            if isinstance(data, dict):
                # Convertir dict a DataFrame si es necesario
                try:
                    df = pd.DataFrame([data])
                    if not df.empty and len(df.columns) >= 5:
                        fixed_data[symbol] = df
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo convertir {symbol} a DataFrame: {e}")
            
            elif isinstance(data, pd.DataFrame):
                if not data.empty and len(data.columns) >= 5:
                    fixed_data[symbol] = data
    
    if fixed_data:
        logger.info(f"✅ Datos de mercado reparados para {len(fixed_data)} símbolos")
        return fixed_data, f"Datos reparados para {len(fixed_data)} símbolos"
    else:
        logger.error("❌ No se pudieron reparar datos de mercado")
        return {}, "No se pudieron reparar datos de mercado"


def get_market_data_summary(market_data: Dict) -> Dict:
    """
    Obtiene un resumen de los datos de mercado para monitoreo.
    
    Args:
        market_data: Datos de mercado completos
        
    Returns:
        Dict con resumen de datos
    """
    summary = {
        'total_symbols': len(market_data),
        'symbols': list(market_data.keys()),
        'data_types': {},
        'latest_prices': {},
        'data_quality': {}
    }
    
    for symbol, data in market_data.items():
        # Tipo de datos
        summary['data_types'][symbol] = str(type(data))
        
        # Precio actual
        current_price = _extract_current_price_safely(symbol, market_data)
        summary['latest_prices'][symbol] = current_price
        
        # Calidad de datos
        if isinstance(data, pd.DataFrame):
            quality = {
                'rows': len(data),
                'columns': len(data.columns),
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_missing_values': not data.isnull().any().any()
            }
        elif isinstance(data, dict):
            quality = {
                'has_required_fields': all(field in data for field in ['open', 'high', 'low', 'close', 'volume']),
                'all_numeric': all(isinstance(data[field], (int, float)) for field in ['open', 'high', 'low', 'close', 'volume'])
            }
        else:
            quality = {'type_error': True}
        
        summary['data_quality'][symbol] = quality
    
    return summary


def validate_data_consistency(market_data: Dict) -> Dict:
    """
    Valida la consistencia de los datos de mercado.
    
    Args:
        market_data: Datos de mercado a validar
        
    Returns:
        Dict con resultados de validación de consistencia
    """
    consistency_results = {
        'valid': True,
        'issues': [],
        'symbol_issues': {}
    }
    
    for symbol, data in market_data.items():
        symbol_issues = []
        
        if isinstance(data, pd.DataFrame):
            # Validar consistencia de precios
            if 'open' in data.columns and 'close' in data.columns:
                for idx, row in data.iterrows():
                    if row['high'] < max(row['open'], row['close']) or row['low'] > min(row['open'], row['close']):
                        symbol_issues.append(f"Fila {idx}: inconsistencia OHLC")
                        consistency_results['valid'] = False
            
            # Validar volumen positivo
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    symbol_issues.append(f"Volumen negativo en {negative_volume} filas")
                    consistency_results['valid'] = False
        
        elif isinstance(data, dict):
            # Validar consistencia de precios en dict
            if all(field in data for field in ['open', 'high', 'low', 'close']):
                if data['high'] < max(data['open'], data['close']) or data['low'] > min(data['open'], data['close']):
                    symbol_issues.append("Inconsistencia OHLC en dict")
                    consistency_results['valid'] = False
            
            # Validar volumen positivo
            if 'volume' in data and data['volume'] < 0:
                symbol_issues.append("Volumen negativo en dict")
                consistency_results['valid'] = False
        
        if symbol_issues:
            consistency_results['symbol_issues'][symbol] = symbol_issues
            consistency_results['issues'].extend([f"{symbol}: {issue}" for issue in symbol_issues])
    
    return consistency_results


def sanitize_market_data(market_data: Dict) -> Dict:
    """
    Sanitiza datos de mercado eliminando valores inválidos.
    
    Args:
        market_data: Datos de mercado crudos
        
    Returns:
        Dict con datos sanitizados
    """
    sanitized_data = {}
    
    for symbol, data in market_data.items():
        try:
            if isinstance(data, pd.DataFrame):
                # Eliminar filas con valores nulos o negativos
                clean_data = data.dropna()
                if 'volume' in clean_data.columns:
                    clean_data = clean_data[clean_data['volume'] >= 0]
                if len(clean_data) > 0:
                    sanitized_data[symbol] = clean_data
            
            elif isinstance(data, dict):
                # Validar y limpiar dict
                if all(isinstance(data.get(field), (int, float)) for field in ['open', 'high', 'low', 'close', 'volume']):
                    if data['volume'] >= 0:
                        sanitized_data[symbol] = data
        
        except Exception as e:
            logger.warning(f"⚠️ Error sanitizando datos para {symbol}: {e}")
    
    return sanitized_data