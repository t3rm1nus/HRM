# 🧹 Script de Limpieza del Sistema HRM

Este script permite limpiar completamente todos los archivos de log, datos temporales, portfolios y cache del sistema HRM antes de iniciar un nuevo proceso de trading.

## 📋 Qué Limpia

### 🗂️ **Directorios Eliminados**
- `logs/` - Todos los logs del sistema
- `test_logs/` - Logs de pruebas
- `backtesting/logs/` - Logs de backtesting
- `results/` - Archivos de resultados
- `data/datos_inferencia/` - Datos de inferencia temporales
- Todos los `__pycache__/` (cache de Python)

### 📄 **Archivos Eliminados**
- `hacienda/posiciones_fifo.json` - Posiciones fiscales
- `portfolio_state_live.json` - Estado del portfolio
- `allocation_tiers_results.json` - Resultados de asignación
- `risk_adjusted_sizing_results.json` - Tamaños ajustados por riesgo
- Todos los archivos `portfolio_*.json`
- Todos los archivos `sentiment_cache_*.json`
- Todos los archivos `sentiment_inference_*.csv`
- **Archivos de análisis (NUEVO)**:
  - `data/historico.csv` - Historial principal del sistema
  - `data/historico.db` - Base de datos del historial
  - `data/portfolio/portfolio_history*.csv` - Historiales de portfolio
  - `data/portfolio/portfolio_state_*.json` - Estados de portfolio
  - `data/logs/trades_history.csv` - Historial de operaciones
- Archivos temporales: `*_cache.json`, `__pycache__`, `.pyc`, `.pyo`, `.tmp`
- `kk.py` - Archivo de debug temporal

### ✅ **Directorios Recreacu**
Después de la limpieza, se recrean automáticamente:
- `logs/`
- `data/datos_inferencia/`
- `results/`

## 🚀 Uso

### **Ejecución Automática (Recomendado)**
El script se ejecuta automáticamente al iniciar `main.py`. Aparecerá algo como:

```
🧹 Running system cleanup before startup...
🧹 Cleaning logs directories...
🧹 Cleaning hcacienda files (tax system)...
🧹 Cleaning portfolio data...
🧹 Cleaning inference and sentiment data...
🧹 Cleaning results data...
🧹 Cleaning temporary files...
📁 Creating fresh directories...
✅ Cleanup completed successfully - 45 files, 12 directories removed
🚀 Starting HRM system
```

### **Ejecución Manual**
También puedes ejecutarlo directamente desde línea de comandos:

```bash
# Limpieza completa
python system_cleanup.py

# Ver qué se eliminaría sin eliminar realmente
python system_cleanup.py --dry-run

# Especificar directorio diferente
python system_cleanup.py --path /ruta/a/tu/proyecto
```

### **Desde dentro de Python**
```python
from system_cleanup import SystemCleanup

# Limpieza completa
cleanup = SystemCleanup()
result = cleanup.perform_full_cleanup()

print(f"Eliminados: {result['deleted_files']} archivos, {result['deleted_dirs']} directorios")
```

## 🔧 Personalización

Puedes modificar la clase `SystemCleanup` para añadir nuevas reglas de limpieza:

1. **Añadir nuevo tipo de limpieza**: Crear método `clean_custom_data()`
2. **Modificar patrones**: Editar listas en los métodos existentes
3. **Cambiar directorios recreados**: Modificar lista `dirs_to_create`

**Ejemplo - Añadir limpieza de datos de ML:**
```python
def clean_ml_cache(self):
    """Limpia cache de modelos ML"""
    logger.info("🧹 Limpiando cache de modelos ML...")

    ml_patterns = ["*.h5", "*.pkl", "model_cache_*.json"]
    for pattern in ml_patterns:
        for file_path in glob.glob(os.path.join(self.base_path, pattern)):
            self.safe_delete_file(file_path, "ML cache:")
```

## ⚡ Seguridad

- ✅ **Elimina archivos de forma segura** (verifica existencia antes de eliminar)
- ✅ **No elimina directorios de git** (`.git/` queda intacto)
- ✅ **No elimina modelos entrenados** (solo cache temporal)
- ✅ **Logs detallados** de todo lo que hace
- ✅ **Modo dry-run** para ver qué eliminaría
- ✅ **Manejo de errores** (continúa aunque algunos archivos fallen)

## 🔍 Debugging

Si algo no se limpia correctamente:

1. **Ver logs**: El script muestra qué elimina y por qué
2. **Modo dry-run**: `python system_cleanup.py --dry-run`
3. **Añadir logging**: El código usa `logger.info()` para todos los pasos

```
</final_file_content>
