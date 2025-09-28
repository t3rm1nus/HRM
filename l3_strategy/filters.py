# filters.py
from core.logging import logger

# Configuraciones específicas para mercados en rango
RANGE_MARKET_SETTINGS = {
    'required_confidence': 0.65,  # Menor confianza requerida (vs 0.8 normal)
    'profit_target': 0.008,       # 0.8% target (más pequeño que targets normales)
    'stop_loss': 0.015,           # 1.5% stop (más ajustado que stops normales)
    'max_position_time': 6        # 6 ciclos máximo en range (vs más tiempo en tendencias)
}

def get_range_filters():
    """
    Retorna los filtros específicos para mercados en rango.
    """
    try:
        logger.info("🎯 Aplicando filtros específicos para mercado en rango")
        return RANGE_MARKET_SETTINGS
    except Exception as e:
        logger.error(f"❌ Error obteniendo filtros de rango: {e}")
        return RANGE_MARKET_SETTINGS  # Retornar defaults en caso de error

def apply_range_filters(signal_data, market_regime):
    """
    Aplica filtros específicos cuando el mercado está en rango.
    """
    try:
        if market_regime != 'range':
            return signal_data  # No modificar si no es rango

        # Aplicar configuraciones de rango
        settings = get_range_filters()

        # Ajustar confianza requerida
        if 'confidence' in signal_data:
            signal_data['adjusted_confidence'] = signal_data['confidence'] >= settings['required_confidence']

        # Ajustar targets
        signal_data['profit_target'] = settings['profit_target']
        signal_data['stop_loss'] = settings['stop_loss']
        signal_data['max_position_time'] = settings['max_position_time']

        logger.info(f"🔧 Filtros de rango aplicados: Confianza >= {settings['required_confidence']}, Target: {settings['profit_target']}, Stop: {settings['stop_loss']}")
        return signal_data

    except Exception as e:
        logger.error(f"❌ Error aplicando filtros de rango: {e}")
        return signal_data
