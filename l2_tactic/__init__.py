from .signal_generator import generar_senales
from .position_sizer import calcular_sizing
from .risk_controls import aplicar_riesgo
from comms.config import SYMBOL  # o SYMBOLS si usas varios pares

def procesar_l2(state):
    ordenes = []
    for activo, target in state["exposicion"].items():
        current = state["portfolio"].get(activo, 0.0)
        delta = round(target - current, 8)  # diferencia en unidades

        if abs(delta) > 1e-8:
            side = "buy" if delta > 0 else "sell"
            ordenes.append({
                "symbol": f"{activo}/USDT",   # ejemplo: BTC → BTC/USDT
                "side": side,
                "amount": abs(delta),         # cantidad en unidades del activo
                "type": "market",             # por ahora solo market
                "price": None                 # reservado si luego usamos limit
            })

    state["ordenes"] = ordenes
    return state