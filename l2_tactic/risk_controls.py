# l2_tactic/risk_controls.py

def aplicar_riesgo(sizing):
    """
    Ajusta el sizing de posiciones según límites de riesgo.
    Retorna un diccionario con los tamaños ajustados.
    """
    sizing_ajustado = {}
    for activo, tamaño in sizing.items():
        # Ejemplo simple: máximo 5% del capital por activo
        max_riesgo = 0.05
        sizing_ajustado[activo] = min(tamaño, max_riesgo)
    return sizing_ajustado
