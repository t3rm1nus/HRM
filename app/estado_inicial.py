# app/estado_inicial.py - Estado de arranque y salida de la capa L3

from datetime import datetime

from comms.config import APAGAR_L3
from core.logging import logger
from system.state_coordinator import StateCoordinator


def crear_salida_l3(regimen: str, tipo_estrategia: str, confianza: float = 0.0) -> dict:
    return {
        'regime': regimen,
        'signal': 'hold',
        'confidence': confianza,
        'strategy_type': tipo_estrategia,
        'timestamp': datetime.utcnow().isoformat(),
    }


def obtener_estado_inicial(state_coordinator: StateCoordinator) -> dict:
    logger.info("🔧 Getting initial state...")
    try:
        estado = state_coordinator.get_state("current")
        logger.info("✅ Initial state obtained")
        return estado
    except Exception as error:
        logger.error(f"❌ Failed to get state: {error}")
        logger.warning("⚠️ Using fallback state")
        return {
            "market_data": {},
            "total_value": 0.0,
            "l3_output": crear_salida_l3("error", "fallback"),
        }


def preparar_salida_l3(estado: dict, components: dict):
    if APAGAR_L3:
        estado["l3_output"] = crear_salida_l3("disabled", "l3_disabled")
    elif "l3_output" not in estado:
        estado["l3_output"] = components.get('l3_output', crear_salida_l3("error", "l3_error"))

    if not estado.get("l3_output"):
        estado["l3_output"] = crear_salida_l3("neutral", "initial", confianza=0.5)
        logger.warning("🔄 Fixed empty l3_output in state")
