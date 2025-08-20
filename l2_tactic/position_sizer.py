# En l2_tactic/position_sizer.py
def calcular_sizing(senales, exposicion):
    """
    senales: dict con {activo: 'comprar'/'vender'/'mantener'}
    exposicion: dict con {activo: float}
    """
    mapping = {
        "comprar": 1,
        "mantener": 0,
        "vender": -1
    }

    sizing = {}
    for asset, senal in senales.items():
        senal_num = mapping.get(str(senal).lower(), 0)  # default 0 si no coincide
        if senal_num > 0:
            sizing[asset] = exposicion.get(asset, 0)
        else:
            sizing[asset] = 0
    return sizing
