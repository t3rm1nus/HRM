from .universe_filter import filtrar_universo

def procesar_l3(state):
    universo = ["BTC", "ETH", "USDT"]
    exposicion = {act: 0.5 for act in universo}

    state["universo"] = universo
    state["exposicion"] = exposicion
    return state


def filtrar_universo(datos_mercado, portfolio_state):
    universo = []
    for asset in portfolio_state:
        # Verifica que el activo exista en datos_mercado y tenga precio positivo
        if asset in datos_mercado and datos_mercado[asset].get("precio", 0) > 0:
            universo.append(asset)
    return universo