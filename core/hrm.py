from typing import Dict, Any

# Adaptación simplificada del pipeline para datos históricos

def ciclo_historico(datos_mercado: Dict[str, float], estado: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Ejecuta un ciclo con datos históricos (sin async). Devuelve el nuevo estado.
	- datos_mercado: mapping {activo: precio}
	- estado: estado acumulado del backtest
	"""
	# Actualiza mercado en el estado
	nuevo_estado = dict(estado)
	nuevo_estado["mercado"] = dict(datos_mercado)

	# Señal trivial: si precio > 0, mantener exposición objetivo si existe, si no mantener
	universo = nuevo_estado.get("universo", list(datos_mercado.keys()))
	exposicion = nuevo_estado.get("exposicion", {a: 0.0 for a in universo})

	# Genera órdenes simples en función de exposición objetivo
	ordenes = []
	portfolio = dict(nuevo_estado.get("portfolio", {}))
	for activo in universo:
		target = exposicion.get(activo, 0.0)
		actual = portfolio.get(activo, 0.0)
		delta = target - actual
		if abs(delta) > 0:
			ordenes.append({"activo": activo, "cantidad": delta})
			portfolio[activo] = actual + delta

	nuevo_estado["ordenes"] = ordenes
	nuevo_estado["portfolio"] = portfolio
	return nuevo_estado