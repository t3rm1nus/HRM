# 🏛️ Módulo Hacienda - Gestión Fiscal Española

Este módulo implementa el seguimiento fiscal completo para criptomonedas según la normativa española, incluyendo el cálculo de ganancias/pérdidas por el método FIFO.

## 📋 Características

- ✅ **Seguimiento automático** de todas las operaciones (compras/ventas)
- ✅ **Cálculo FIFO** para ganancias/pérdidas (primera entrada, primera salida)
- ✅ **Informes fiscales anuales** con base imponible
- ✅ **Exportación de datos** en múltiples formatos (CSV, JSON, Excel)
- ✅ **Integración automática** con el portfolio manager
- ✅ **Cumplimiento normativo** español para declaración de impuestos

## 📁 Estructura de Archivos

```
hacienda/
├── __init__.py              # Inicialización del módulo
├── tax_tracker.py           # Motor principal de seguimiento fiscal
├── tax_utils.py             # Utilidades y herramientas fiscales
├── README.md                # Esta documentación
├── operaciones.csv          # Historial de todas las operaciones
├── posiciones_fifo.json     # Posiciones FIFO actuales
├── ganancias_realizadas.csv # Ganancias/pérdidas realizadas
├── informe_fiscal_YYYY.json # Informe fiscal anual
└── declaracion_impuestos_YYYY.csv # Datos para declaración
```

## 🚀 Uso Básico

### 1. Inicialización Automática

El módulo se integra automáticamente con el portfolio manager. No requiere configuración adicional.

### 2. Generar Informe Fiscal

```python
from hacienda.tax_utils import TaxUtils

# Inicializar utilidades fiscales
tax_utils = TaxUtils()

# Mostrar resumen fiscal del año actual
tax_utils.show_tax_summary()

# Generar informe completo
report = tax_utils.generate_annual_tax_report(2024)
```

### 3. Exportar Datos para Declaración

```python
# Exportar en formato CSV (recomendado para Excel)
csv_file = tax_utils.export_tax_data(2024, format="csv")

# Exportar en formato Excel con múltiples hojas
excel_file = tax_utils.export_tax_data(2024, format="excel")
```

### 4. Uso desde Línea de Comandos

```bash
# Mostrar resumen fiscal
python -m hacienda.tax_utils --action summary --year 2024

# Exportar datos en CSV
python -m hacienda.tax_utils --action export --year 2024 --format csv

# Generar informe completo
python -m hacienda.tax_utils --action report --year 2024
```

## 📊 Datos Registrados

### Operaciones (`operaciones.csv`)
- ID de operación único
- Fecha y hora exacta
- Símbolo de criptomoneda
- Tipo de operación (Compra/Venta)
- Cantidad y precio
- Valor total y neto
- Comisión del exchange
- Año fiscal

### Ganancias Realizadas (`ganancias_realizadas.csv`)
- Fecha de compra y venta
- Cantidad vendida
- Precio de compra y venta
- Coste base y valor de venta
- Ganancia/pérdida calculada
- Días de tenencia

### Posiciones FIFO (`posiciones_fifo.json`)
- Posiciones abiertas por criptomoneda
- Cantidad restante por posición
- Precio promedio ponderado
- Historial de compras no vendidas

## 💰 Cálculo de Ganancias/Pérdidas

### Método FIFO (First In, First Out)
1. **Compra**: Se añade como nueva posición en la cola
2. **Venta**: Se vende primero la posición más antigua
3. **Cálculo**: Ganancia = Precio_venta - Precio_compra
4. **Impuestos**: Según período de tenencia (<1 año o >1 año)

### Ejemplo Práctico:
```
Compra 1: 1 BTC @ $50,000 → Posición A
Compra 2: 1 BTC @ $60,000 → Posición B
Venta: 1.5 BTC @ $55,000

Resultado:
- Vende 1 BTC de Posición A: +$5,000 ganancia
- Vende 0.5 BTC de Posición B: -$2,500 pérdida
- Ganancia neta: +$2,500
```

## 📋 Información Fiscal Española

### Período de Tenencia
- **Corto plazo** (< 1 año): IRPF general (19-47%)
- **Largo plazo** (≥ 1 año): IRPF reducido (19-26%)

### Declaración de Impuestos
- **Modelo 100**: IRPF - Base imponible del ahorro
- **Modelo 720**: Bienes en el extranjero (>50,000€)
- **Plazo**: Abril-Junio del año siguiente

### Compensación de Pérdidas
- Pérdidas se compensan con ganancias del mismo año
- Límite: 25% de la base imponible general
- Pérdidas no compensadas se arrastran a años siguientes

## ⚠️ Importante

### Descargo de Responsabilidad
- Esta herramienta calcula ganancias/pérdidas según normativa actual
- Los datos son para ayuda en la declaración de impuestos
- **SIEMPRE consulta con tu asesor fiscal o AEAT**
- La interpretación de la normativa puede cambiar

### Recomendaciones
1. **Revisa los cálculos** antes de usarlos en tu declaración
2. **Guarda backups** de todos los archivos generados
3. **Actualiza regularmente** los informes fiscales
4. **Consulta cambios normativos** en la web de la AEAT

## 🔧 Configuración Avanzada

### Personalizar Ubicación de Archivos
```python
from hacienda.tax_tracker import TaxTracker

# Usar directorio personalizado
tracker = TaxTracker(hacienda_dir="/ruta/personalizada/hacienda")
```

### Integración Manual
```python
from hacienda.tax_tracker import TaxTracker

# Registrar operación manualmente
tracker = TaxTracker()
operation = {
    'symbol': 'BTCUSDT',
    'side': 'buy',
    'quantity': 0.1,
    'filled_quantity': 0.1,
    'price': 50000.0,
    'filled_price': 50000.0,
    'commission': 5.0,
    'status': 'filled'
}
tracker.record_operation(operation, exchange="Binance")
```

## 📞 Soporte

Para soporte técnico o consultas sobre el módulo Hacienda:
- Revisa los logs en `logs/` para diagnóstico
- Verifica la integridad de los archivos CSV/JSON
- Consulta la documentación de la AEAT para cambios normativos

## 🔄 Actualizaciones

El módulo se actualiza automáticamente con:
- Cambios en la normativa fiscal española
- Mejoras en el cálculo FIFO
- Nuevos formatos de exportación
- Corrección de bugs y optimizaciones

---

**⚖️ Recuerda**: Esta herramienta es un auxiliar para la gestión fiscal, no sustituye el asesoramiento profesional.
