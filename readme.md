# 🔱 HRM — Hierarchical Reasoning Model para Trading Algorítmico
**Estado: PRODUCCIÓN** · **Lenguaje:** Python 3.10+ · **Dominio:** Cripto Trading · **Arquitectura:** L3 Estratégico + L2 Táctico + L1 Operacional

## 🧭 TL;DR
**HRM - HIERARCHICAL REASONING MODEL - SISTEMA DE TRADING MULTI-ESTRATEGIA CON 3 PATHS**

HRM es un sistema de trading algorítmico **multi-estrategia con 3 paths operativos** que opera con BTC y ETH en Binance Spot. Ha sido **reformulado completamente** para eliminar contradicciones estratégicas y hacer que la jerarquía de decisión sea clara. El sistema combina **clasificación de regímenes L3**, **modelos FinRL especializados**, **fuerza técnica avanzada**, **convergencia L1+L2** y **ejecución determinista** con protección HARDCORE.

---

## 🚀 **ULTIMOS CAMBIOS Y MEJORAS 2025 - LEGACY CODE CLEANUP COMPLETED**

### ✅ **1. LEGACY CODE CLEANUP - OBSERVATION BUILDERS REFACTORIZADO**
**Fecha:** Octubre 2025
**Archivo:** `l2_tactic/observation_builders.py`
**Impacto:** Arquitectura totalmente modularizada

#### 🎯 **Cambios Implementados:**
- ✅ **Eliminación de funciones legacy:** `build_legacy_observation()` y `build_gemini_obs()` removidas
- ✅ **Sistema modular moderno:** Solo funciones optimizadas para producción
- ✅ **Determinismo mejorado:** Observaciones consistentes para modelos FinRL
- ✅ **Performance optimizado:** Reducción de latencia en generación

#### 🔄 **2. SENTIMENT UPDATE INTERVAL CORRECTED**
**Fecha:** Octubre 2025
**Archivo:** `main.py` línea 386
**Impacto:** Timing corregido para expiración de cache BERT

- ✅ **Intervalo anterior:** 42 ciclos (~40 minutos) ⏰❌
- ✅ **Intervalo corregido:** 2160 ciclos (~6 horas) ✨✅
- ✅ **Alineación perfecta:** Sincronizado con BERT cache expiration
- ✅ **Optimización de recursos:** 51x reducción en llamadas API Reddit/News

#### 📊 **3. JERARQUÍA DE DECISIÓN CLARA Y MODULARIZACIÓN COMPLETA**
**Estado:** ✅ **COMPLETAMENTE IMPLEMENTADO**

- ✅ **L3 domina estratégicamente:** Override automático de señales contradictorias
- ✅ **Stop-loss inteligentes:** Cálculo dinámico basado en volatilidad
- ✅ **Auto-aprendizaje con protección:** 9 capas anti-overfitting activas
- ✅ **Sistema HARDCORE de protección:** Sincronización completa con exchange
- ✅ **Arquitectura modular L2:** FinRL processors especializados por modelo

### 🚀 **IMPLEMENTACIÓN COMPLETA DEL SISTEMA HRM 2025**
**Estado de Operatividad:** ✅ **PRODUCCIÓN LISTA**

| Componente | Estado | Características |
|------------|--------|----------------|
| **FinRL Modular** | ✅ Operativo | 6 modelos soportados con detección automática |
| **L3 Strategy** | ✅ Completo | 5 modelos IA con regime detection y sentiment |
| **L2 Tactic** | ✅ Modular | Arquitectura limpia con 10 mejoras Crushing |
| **L1 Operational** | ✅ Optimizado | Gestión de liquidez y validaciones avanzadas |
| **Auto-Learning** | ✅ Autonomo | 9 capas de protección anti-overfitting |
| **HARDCORE Safety** | ✅ Ultra-seguro | Sincronización real con exchange |

### 📈 **IMPACTO TOTAL DE LAS MEJORAS**
- ✅ **Decision Making:** Jerarquía clara L3 → L2 → L1
- ✅ **Risk Management:** Stop-loss dinámicos + correlación inteligente
- ✅ **Performance:** Auto-aprendizaje continuo con evolución de modelos
- ✅ **Safety:** Sistema ultra-seguro contra fallos de conectividad
- ✅ **Scalability:** Arquitectura modular preparada para más activos

### 🎯 **OBJETIVO ALCANZADO**
**HRM ahora es un sistema de trading algorítmico de nivel institucional con:**
- 🛡️ Protección extrema de capital
- 🤖 Aprendizaje continuo autónomo
- ⚡ Arquitectura modular escalable
- 📊 14 modelos IA operativos
- 🎯 Jerarquía de decisión clara y determinista

**✨ Sistema HRM 2025: LEGACY CODE CLEANED & FULLY MODULARIZED** 🚀

---

## ✅ **SISTEMA HEREDADO DEL CONTRADICCIÓN ARQUITURAL CRÍTICA** ❌
**🚨 ANTES:** Sistema híbrido con contradicciones
- ❌ **L2 mantenía lógica de mean-reversion** (RSI <30 compra)
- ❌ **L3 detectaba regímenes pero no dominaba decisiones**
- ❌ **Ejecución: ~4.4% con win rate casi nulo**

## 🎯 **NUEVO SISTEMA: JERARQUÍA DE DECISIÓN CLARA CON 3 PATHS**
**🚀 AHORA:** Arquitectura multi-estrategia coherente
- ✅ **Mean-reversion completamente eliminado** (no más RSI <30)
- ✅ **Jerarquía de decisión clara** para evitar contradicciones
- ✅ **Objetivo:** Ejecutar >30% con win rate >55%

## ✅ **SISTEMA HEREDADO DEL CONTRADICCIÓN ARQUITURAL CRÍTICA** ❌
**🚨 ANTES:** Sistema híbrido con contradicciones
- ❌ **L2 mantenía lógica de mean-reversion** (RSI <30 compra)
- ❌ **L3 detectaba regímenes pero no dominaba decisiones**
- ❌ **Ejecución: ~4.4% con win rate casi nulo**

## 🎯 **NUEVO SISTEMA: JERARQUÍA DE DECISIÓN CLARA CON 3 PATHS**
**🚀 AHORA:** Arquitectura multi-estrategia coherente
- ✅ **Mean-reversion completamente eliminado** (no más RSI <30)
- ✅ **Jerarquía de decisión clara** para evitar contradicciones
- ✅ **Objetivo:** Ejecutar >30% con win rate >55%

### ✅ **ESTADO ACTUAL: OPERATIVO CON NUEVO PLAN**
- ✅ **Carga móvil a Binance Spot** (modo LIVE y TESTNET)
- ✅ **Modo simulado con 3000 USDT** para testing seguro
- ✅ **Señales cada 8-10 segundos** con indicadores trend-following
- ✅ **Modelos IA especializados** en trend-following (no más mean-reversion)
- ✅ **Sistema trend-following L3 dominante** con override automático
- ✅ **Gestión de portfolio automática** con tracking en CSV
- ✅ **Logging centralizado** en core/logging.py (formato estandarizado)
- ✅ **Stop-Loss y Take-Profit dinámicos** por señal
- ✅ **Costos reales de trading** (comisiones 0.1% Binance)
- ✅ **9 modelos AI operativos** (3 L1 + 1 L2 + 5 L3)
- ✅ **Análisis de sentimiento en tiempo real** (Reddit + News API)

## 🛡️ SISTEMA DE PROTECCIÓN "HARDCORE" - PRODUCCIÓN ULTRA-SEGURO

**🔴 CRÍTICO PARA OPERACIONES REALES:** HRM incluye un sistema de protección multi-nivel diseñado para entornos de producción extremos donde fallos de conectividad o energía pueden causar pérdidas catastróficas.

### 🚨 **PROBLEMAS RESUELTOS**
- **❌ Stop-Loss NO guardados:** Antes solo cálculos locales, posiciones desprotegidas
- **❌ Sin sincronización:** Sistema no verificaba posiciones reales en exchange
- **❌ Pérdidas por crashes:** Reinicio perdía estado y dejaba posiciones expuestas
- **❌ Desincronización:** Estado local ≠ estado real del exchange

### ✅ **SOLUCIONES IMPLEMENTADAS**

#### 🛡️ **1. STOP-LOSS REALES EN BINANCE**
```python
# STOP-LOSS colocados REALMENTE en el exchange
sl_order = await binance_client.place_stop_loss_order(
    symbol="BTCUSDT",
    side="SELL",
    quantity=0.001,
    stop_price=45000.0,  # Precio real de activación
    limit_price=44900.0  # Precio de ejecución
)
```
- **Modo LIVE:** Órdenes STOP_LOSS colocadas en Binance Spot real
- **Modo TESTNET:** Órdenes simuladas pero con lógica idéntica
- **Protección 24/7:** Stop-loss persisten aunque el sistema se caiga

#### 🔄 **2. SINCRONIZACIÓN OBLIGATORIA AL INICIO**
```python
# CRÍTICO: Verificación de estado real al startup
sync_success = await portfolio_manager.sync_with_exchange()
if sync_success:
    logger.info("✅ Portfolio sincronizado con Binance real")
    # Sistema continúa con posiciones correctas
else:
    logger.error("❌ FALLO DE SINCRONIZACIÓN - ABORTAR OPERACIÓN")
```
- **Verificación automática:** Compara estado local vs exchange real
- **Detección de discrepancias:** Alerta si hay diferencias significativas
- **Corrección automática:** Actualiza estado local con datos reales

#### 🚨 **3. DETECCIÓN DE DESINCRONIZACIÓN**
```python
# Monitoreo continuo de integridad
btc_diff = abs(local_btc - exchange_btc)
if btc_diff > 0.0001:
    logger.warning("🚨 DESINCRONIZACIÓN BTC: Local vs Exchange")
    # Corrección automática o alerta crítica
```
- **Monitoreo en tiempo real:** Comparación continua local vs exchange
- **Alertas automáticas:** Notificación inmediata de discrepancias
- **Corrección automática:** Re-sincronización cuando se detecta

#### 🔌 **4. RECUPERACIÓN TRAS FALLOS DE CONECTIVIDAD**
```python
# Escenario: Se va la luz → Vuelve la conexión
# 1. Sistema reinicia automáticamente
# 2. sync_with_exchange() lee posiciones reales
# 3. Stop-loss orders siguen activas en Binance
# 4. Sistema continúa con estado correcto
```
- **Recuperación automática:** Sistema se re-sincroniza tras fallos
- **Estado consistente:** Posiciones y stops preservados en exchange
- **Continuidad operativa:** Trading continúa sin intervención manual

### 🎯 **ARQUITECTURA DE PROTECCIÓN**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SISTEMA HRM   │    │    BINANCE      │    │   POSICIONES    │
│                 │    │    EXCHANGE     │    │     REALES      │
│  ┌─────────┐    │    │                 │    │                 │
│  │ STOP-   │◄───┼────┤ STOP-LOSS       │    │  🛡️ PROTEGIDAS  │
│  │ LOSS    │    │    │ REALES          │    │                 │
│  │ LOCAL   │    │    │                 │    │                 │
│  └─────────┘    │    └─────────────────┘    └─────────────────┘
│                 │              ▲
│  ┌─────────┐    │              │
│  │ SINCRONIZ│◄──┼──────────────┘
│  │ ZACIÓN   │    │    VERIFICACIÓN AUTOMÁTICA
│  └─────────┘    │    AL INICIO Y DURANTE OPERACIÓN
└─────────────────┘
```

### 📊 **ESTADOS DE PROTECCIÓN**

| Estado | Descripción | Acción |
|--------|-------------|--------|
| **🟢 SINCRONIZADO** | Estado local = Exchange real | Operación normal |
| **🟡 DESINCRONIZADO** | Diferencias detectadas | Re-sincronización automática |
| **🔴 CRÍTICO** | Fallo de sincronización | Alerta + Modo seguro |
| **⚫ OFFLINE** | Sin conexión | Stop-loss en exchange activos |

### ⚙️ **CONFIGURACIÓN PARA PRODUCCIÓN**

```bash
# Variables críticas para modo HARDCORE
export BINANCE_MODE=LIVE
export USE_TESTNET=false
export HRM_HARDCORE_MODE=true  # Activa protecciones máximas
export HRM_SYNC_ON_STARTUP=true  # Sincronización obligatoria
export HRM_STOPLOSS_REAL=true  # Stop-loss reales en exchange

# Monitoreo adicional
export HRM_HEALTH_CHECK_INTERVAL=30  # Segundos
export HRM_MAX_DESYNC_TOLERANCE=0.001  # 0.1% máximo desincronización
```

### 🚨 **PROTOCOLOS DE SEGURIDAD**

1. **Inicio del Sistema:**
   - Verificación de conectividad con Binance
   - Sincronización completa de posiciones
   - Validación de stop-loss existentes
   - Solo continúa si sincronización exitosa

2. **Durante Operación:**
   - Monitoreo continuo de estado vs exchange
   - Re-sincronización automática cada 5 minutos
   - Alertas inmediatas por desincronización

3. **Tras Fallos:**
   - Reinicio automático con verificación completa
   - Recuperación de estado desde exchange
   - Validación de integridad antes de continuar

### 🎯 **VENTAJAS DEL SISTEMA HARDCORE**

- **🛡️ Protección 24/7:** Stop-loss persisten aunque el sistema falle
- **🔄 Recuperación automática:** Sin intervención manual tras fallos
- **📊 Transparencia total:** Estado real siempre visible y verificable
- **⚡ Continuidad operativa:** Trading continúa tras desconexiones
- **🚨 Alertas proactivas:** Detección inmediata de problemas

**El sistema HRM ahora es un entorno de producción ultra-seguro donde fallos de conectividad o energía NO resultan en pérdidas catastróficas.**

## 🎛️ **MODOS DE OPERACIÓN**

| Modo | Descripción | Activación |
|------|-------------|------------|
| **PAPER** | Simulación completa sin conexión real | `USE_TESTNET=true` |
| **LIVE** | Ejecución real en Binance Spot (requiere claves API) | `USE_TESTNET=false` |
| **REPLAY** | Reproducción con datasets históricos | Configuración adicional |

### ⚡ **ACTIVAR MODO LIVE**
```bash
export BINANCE_MODE=LIVE
export USE_TESTNET=false
export BINANCE_API_KEY=your_real_key
export BINANCE_API_SECRET=your_real_secret
python main.py
```

## 🎯 **OPERATING MODES (HRM_PATH_MODE)**

The HRM system supports three distinct operating modes controlled by the `HRM_PATH_MODE` environment variable. These modes determine how signals are processed, validated, and executed across the L2 and L1 layers.

### Modes Overview

| Mode | Description | Signal Source | Validation Rules |
|------|-------------|---------------|-----------------|
| **PATH1** | Pure Trend-Following | `path1_pure_trend_following` | No restrictions |
| **PATH2** | Hybrid Intelligent | `path2_*` sources | Contra-allocation limits (20%) |
| **PATH3** | Full L3 Dominance | `path3_full_l3_dominance` | **L3 sources ONLY** (blocks others) |

### 🎯 **PATH1: Pure Trend-Following - MACRO-ONLY STRATEGY**
**Mode:** Pure trend-following dominated by L3 regime analysis
```bash
export HRM_PATH_MODE=PATH1
```

#### 📈 **Strategy Overview**
- **Primary Driver:** L3 regime detection (Bull/Bear/Neutral markets)
- **Signal Source:** Regime classification only - ignores L1/L2 technical signals
- **Approach:** Pure macro-driven trading following market regime trends
- **Risk Level:** Medium (regime changes can be sudden but well-validated)
- **Core Strategy:** Allocates capital based purely on market regime without technical validation

#### 🎯 **Operating Plan**
1. **L3 Regime Analysis:** Classify current market regime using ML ensemble (Random Forest + Gradient Boosting)
2. **Position Strategy:**
   - **Bull Regime:** BUY BTC/ETH, favor risk assets (60% BTC, 30% ETH, 10% CASH)
   - **Bear Regime:** SELL BTC/ETH, favor cash preservation (10% BTC, 5% ETH, 85% CASH)
   - **Neutral Regime:** HOLD current positions, balanced allocation (40% BTC, 30% ETH, 30% CASH)
3. **Entry/Exit Rules:** Pure regime-based, no technical confirmation needed
4. **Rebalancing:** Automatic monthly rebalancing to maintain target allocations

#### 🔧 **Technical Implementation - Regime Detection**
- **Models Used:** Ensemble of 5 ML models (Optuna optimized hyperparameters)
- **Features:** RSI, MACD, Volume Analysis, Volatility metrics, Sentiment scores
- **Classification:** Bull/Bear/Neutral regimes based on 6-month historical patterns
- **Update Frequency:** Every 8-9 minutes (50 cycles)

#### 🛡️ **Risk Controls**
- No technical validation required (pure regime faith)
- Standard stop-loss at 3% per position
- Maximum drawdown limit: 12%
- No contra-trend positioning allowed

#### 📊 **Expected Performance**
- **Bull Markets:** Strong trend-following performance
- **Bear Markets:** Conservative cash preservation
- **Choppy Markets:** May underperform due to holding through noise
- **Best For:** Strong trending periods, institutional macro traders

### 🔄 **PATH2: Hybrid Intelligent - BALANCED MULTI-SIGNAL**
**Mode:** Intelligent combination with sophisticated risk management
```bash
export HRM_PATH_MODE=PATH2
```

#### 🎛️ **Strategy Overview**
- **Primary Driver:** Ensemble of L1+L2+L3 signals with conflict resolution
- **Signal Sources:** Technical (L1), Tactical (L2), Strategic (L3) all contribute
- **Approach:** Intelligent blending with contra-trend risk controls
- **Risk Level:** Medium-High (benefits from multiple perspectives but more complex)

#### 🎯 **Operating Plan**
1. **Multi-Level Signal Generation:**
   - **L1:** Technical signals (RSI, MACD, volume analysis)
   - **L2:** Tactical combination of L1 signals with ML models
   - **L3:** Strategic regime context

2. **Intelligent Voting System:**
   - Signals weighted by confidence scores
   - L3 has veto power but L1/L2 can override within limits
   - Minimum 60% agreement required for execution

3. **Position Strategy:**
   - **Bull Alignment:** Full position sizing (up to 2.5x base size)
   - **Partial Agreement:** Reduced sizing (0.6x - 1.8x base size)
   - **Strong Disagreement:** No position (regime veto)
   - **Contra-Trend Limits:** Maximum 20% of capital in opposite direction

4. **Dynamic Adjustment:**
   - Real-time position sizing based on convergence scores
   - Automatic reduction when L1 disagrees with L3

#### 🛡️ **Risk Controls**
- Contra-trend allocation capped at 20%
- Technical strength validation for large positions
- Enhanced monitoring of signal divergence
- Emergency reduction if 70%+ signals conflict

#### 📊 **Expected Performance**
- **Strong Trends:** Excellent performance with conviction sizing
- **Choppy Markets:** Good performance avoiding whipsaws
- **Mixed Signals:** Conservative approach prevents major mistakes
- **Best For:** Most market conditions, sophisticated retail/institutional

### 🛡️ **PATH3: Full L3 Dominance**
**Mode:** L3 dominates completely, blocks competing signals
```bash
export HRM_PATH_MODE=PATH3
```
- **Strategy:** L3 has 100% control - blocks any non-L3 trend-following signals
- **Signals:** **ONLY `path3_full_l3_dominance` signals allowed**
- **Validation:** **STRICT** - Any order not from L3 sources is **REJECTED**
- **Use Case:** Maximum risk control, regime-driven with iron discipline

### ⚙️ **Configuration**
```python
# In core/config.py
HRM_PATH_MODE = "PATH3"  # Set your preferred mode: PATH1, PATH2, PATH3
```

### 🔒 **Path-Specific Order Validation**

**In PATH3 mode, L1 order_manager.py enforces strict validation:**

```python
def validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
    # PATH3 VALIDATION: Only allow L3 trend-following orders
    if HRM_PATH_MODE == "PATH3":
        if signal_source != "path3_full_l3_dominance":
            return {"valid": False, "reason": "Non-L3 orders blocked in PATH3"}
```

**Validation Results:**
- ✅ **PATH3 L3 signals** → **ALLOWED** → Clean execution
- 🚫 **PATH3 non-L3 signals** → **BLOCKED** → `validate_order()` rejects order
- ✅ **PATH1/PATH2 signals** → **ALLOWED** → No restrictions

### 📊 **Mode Comparison**

| Aspect | PATH1 | PATH2 | PATH3 |
|--------|-------|-------|-------|
| **Signal Diversity** | L3 only | L1+L2+L3 balanced | L3 dominant |
| **Risk Level** | Medium | High (controlled) | **Low** |
| **Contra-Trading** | Not allowed | Limited (20%) | **None allowed** |
| **Validation Strictness** | None | Moderate | **Maximum** |
| **Use Case** | Trend following | Intelligent hybrid | Risk-averse |

### 🔄 **Runtime Switching**
```bash
# Switch modes at runtime (requires restart)
export HRM_PATH_MODE=PATH3
python main.py

# Different modes for different market conditions:
# - PATH1: Strong trending markets
# - PATH2: Sideways/choppy markets
# - PATH3: High uncertainty/volatile markets
```

### 🎯 **Benefits**

1. **🎛️ Operational Flexibility**: Adapt strategy to market conditions
2. **🛡️ Enhanced Safety**: PATH3 blocks potentially risky signals + auto-recovery
3. **📊 Strategy Optimization**: Tailored approaches per market regime
4. **⚡ Controlled Risk**: Path-specific validation prevents unwanted exposures

**The HRM Path Mode system provides iron-clad control over signal execution, ensuring your trading strategy matches your risk tolerance and market assessment.**

---

## 📋 **EXAMPLES: Config.yml with New PATH3 Safety Features**

### 🛡️ **Complete PATH3 Production Configuration**

```yaml
# HRM Production Configuration - PATH3 Auto-Rebalance Mode
# Path: configs/production_path3.yml

# Core Configuration
hr_path_mode: "PATH3"
binance_testnet: false
hardcore_mode: true

# ================================
# PORTFOLIO LIMITS (Safety Features)
# ================================
portfolio_limits:
  # Core Safety - Circuit Breaker
  enable_auto_rebalance: true

  # Checksums & Verification
  rebalance_checksum_verification: true

  # Dry Run Mode (set to false for live trading)
  rebalance_dry_run_enabled: false

  # Fees & Slippage Protection
  rebalance_fees_buffer: 1.01  # 1% buffer for trading fees

  # Cooldown & Lock Management
  rebalance_cooldown_extended: 300  # 5 minutes between operations
  rebalance_min_order_check: true   # Validate USDT before buys

  # Auto-Rebalance Trigger Thresholds
  rebalance_trigger_stoploss_count: 5     # Trigger after 5 stop-losses
  rebalance_max_deviation: 0.10           # Trigger on 10% L3 deviation
  rebalance_min_usdt_reserve: 500         # Min USDT reserve

  # Standard Portfolio Limits (unchanged)
  max_portfolio_exposure_btc: 0.40
  max_portfolio_exposure_eth: 0.40
  max_position_size_usdt: 1200
  min_usdt_reserve: 0.20
  rebalance_threshold: 0.15
  rebalance_trigger_threshold: 5
  rebalance_interval: 60
  rebalance_min_amount: 500
  rotation_amount: 0.25
  min_account_balance_usdt: 500

# ================================
# ROLLBACK INSTRUCTIONS
# ================================

# To disable auto-rebalance (emergency rollback):
# export ENABLE_AUTO_REBALANCE=false

# To enable dry-run mode for testing:
# export REBALANCE_DRY_RUN_ENABLED=true

# To extend cooldown period (reduce frequency):
# export REBALANCE_COOLDOWN_EXTENDED=600  # 10 minutes

# To disable checksum verification:
# export REBALANCE_CHECKSUM_VERIFICATION=false

# ================================
# L3 ALLOCATION TARGETS
# ================================
l3_allocation_targets:
  BTC: 0.40    # 40% BTC
  ETH: 0.30    # 30% ETH
  CASH: 0.30   # 30% USDT reserve

# ================================
# DEPLOYMENT SEQUENCE
# ================================

# Phase 1: Dry Run Testing (1 week)
# rebalance_dry_run_enabled: true
# enable_auto_rebalance: true
# Test simulation logs in: logs/path3_rebalance_audit_*.jsonl

# Phase 2: Live Operation (reduced frequency)
# rebalance_cooldown_extended: 600  # 10 min cooldown

# Phase 3: Full Production
# rebalance_dry_run_enabled: false
# rebalance_cooldown_extended: 300   # 5 min cooldown

# ================================
# MONITORING CHECKS
# ================================

# Audit logs location:
# logs/path3_rebalance_audit_YYYYMMDD.jsonl

# Critical alerts to monitor:
# - "PATH3 REBALANCE TRIGGERED"
# - "DRY RUN FAILED"
# - "BUY ORDER REJECTED"
# - "PORTFOLIO_AUDIT_PRE_REBALANCE"

# Emergency stop:
# export HRM_PATH_MODE=PATH2  # Switch to hybrid mode
```

### 🔧 **Development/Test Configuration**

```yaml
# HRM Development Configuration - PATH3 Safe Mode
# Path: configs/dev_path3.yml

hr_path_mode: "PATH3"
binance_testnet: true  # Always use testnet for development

portfolio_limits:
  # Safety Features - Conservative Settings
  enable_auto_rebalance: true

  # Checksums Enabled for Audit Testing
  rebalance_checksum_verification: true

  # Always Dry Run for Development
  rebalance_dry_run_enabled: true

  # Conservative Timings
  rebalance_fees_buffer: 1.01
  rebalance_cooldown_extended: 60    # 1 minute for testing
  rebalance_min_order_check: true

  # Sensitive Triggers for Development Testing
  rebalance_trigger_stoploss_count: 2
  rebalance_max_deviation: 0.05      # 5% deviation trigger
  rebalance_min_usdt_reserve: 100    # Lower threshold for testing

  # Reduced Position Limits for Development
  max_portfolio_exposure_btc: 0.20   # Conservative
  max_portfolio_exposure_eth: 0.15
  max_position_size_usdt: 500        # Smaller positions

# ================================
# DEVELOPMENT MONITORING
# ================================

# In development mode, monitor these key logs:
# 1. Dry-run simulation results
# 2. Checksum verification logs
# 3. Trigger condition alerts
# 4. Order rejection reasons

# Test with artificial triggers:
# - Force stop-loss: Manually create stop-loss events
# - Force deviation: Adjust L3 allocations manually
# - Test USDT reserve: Spend down USDT balance
```

### � **Migration from PATH2 to PATH3**

```yaml
# Migration Guide: PATH2 → PATH3

# Step 1: Enable PATH3 with Safety Features (1 week)
hr_path_mode: "PATH2"  # Keep current mode
portfolio_limits:
  enable_auto_rebalance: true
  rebalance_dry_run_enabled: true  # Test auto-rebalance safely
  rebalance_checksum_verification: true

# Step 2: Switch to PATH3 (after testing)
hr_path_mode: "PATH3"  # Switch mode
portfolio_limits:
  rebalance_dry_run_enabled: true   # Keep dry-run for safety
  rebalance_cooldown_extended: 300  # Conservative timing

# Step 3: Full PATH3 Production (after 1 week monitoring)
portfolio_limits:
  rebalance_dry_run_enabled: false  # Enable live operations
  rebalance_checksum_verification: true  # Keep audit trails
```

### 📊 **Performance Monitoring Queries**

```sql
-- Query audit logs for PATH3 operations
SELECT * FROM audit_logs
WHERE phase LIKE 'PORTFOLIO_AUDIT_%'
  AND timestamp >= '2025-01-01'
ORDER BY timestamp DESC;

-- Check auto-rebalance frequency
SELECT COUNT(*) as rebalance_count,
       DATE(timestamp) as date
FROM audit_logs
WHERE trigger_conditions LIKE '%stop_loss%'
   OR trigger_conditions LIKE '%l3_deviation%'
GROUP BY DATE(timestamp);

-- Monitor dry-run vs live operations
SELECT phase,
       COUNT(*) as operation_count,
       AVG(estimated_sell_value) as avg_sell_value,
       AVG(estimated_buy_value) as avg_buy_value
FROM audit_logs
WHERE plan IS NOT NULL
GROUP BY phase;
```

### 🎯 **Key Configuration Decisions**

**When to use dry-run mode:**
- New PATH3 deployments
- After parameter changes
- During market volatility
- For testing new L3 allocation targets

**Cooldown timing recommendations:**
- Development: 60 seconds
- Testing: 300 seconds (5 min)
- Production low-risk: 300 seconds (5 min)
- Production high-risk: 600 seconds (10 min)

**Fees buffer based on trading volume:**
- Low volume: 1.005 (0.5% buffer)
- Medium volume: 1.01 (1% buffer)
- High volume: 1.015 (1.5% buffer)

- **Beneficio**: Mejora significativa en calidad de señales
- **Estado**: ✅ **OPERATIVO** - Circuit breakers y multiplicadores dinámicos

#### 🔧 **8. Integration and Testing** ✅ COMPLETADO
- **Funcionalidad**: Integración completa de todos los componentes
- **Testing**: Tests exhaustivos para cada mejora implementada
- **Logging**: Sistema de logging avanzado para todas las nuevas features
- **Beneficio**: Sistema robusto y trazable con monitoreo completo
- **Estado**: ✅ **OPERATIVO** - Pipeline unificado funcionando

#### ⚙️ **9. Configuration and Calibration** ✅ COMPLETADO
- **Funcionalidad**: Configuración completa para todos los nuevos parámetros
- **Calibración**: Sistema de calibración dinámica en tiempo real
- **Monitoreo**: Dashboards para seguimiento de nuevas métricas
- **Beneficio**: Sistema altamente configurable y adaptable
- **Estado**: ✅ **OPERATIVO** - Parámetros ajustables sin downtime

#### 🛡️ **10. Safety and Risk Controls** ✅ COMPLETADO
- **Funcionalidad**: Controles de seguridad multi-nivel con circuit breakers
- **Validación**: Validación exhaustiva de todas las entradas
- **Rollout**: Implementación gradual con fases de seguridad
- **Beneficio**: Protección extrema contra fallos y condiciones adversas
- **Estado**: ✅ **OPERATIVO** - Múltiples capas de protección activas

### 📊 **IMPACTO DE LAS 10 MEJORAS IMPLEMENTADAS**

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Stop-Loss** | Básico | Dinámico inteligente | ✅ Protección superior |
| **Position Sizing** | Fijo | Basado en calidad | ✅ +150% para señales premium |
| **Profit Taking** | Simple | Multi-nivel escalonado | ✅ Captura progresiva |
| **BTC/ETH Sync** | Independiente | Correlacionado inteligente | ✅ Riesgo reducido |
| **Portfolio Mgmt** | Manual | Auto-rebalanceo | ✅ Eficiencia capital |
| **Risk Appetite** | Estático | Dinámico adaptativo | ✅ Adaptabilidad |
| **Convergence** | Ignorada | Multiplicadores dinámicos | ✅ Calidad superior |
| **Integration** | Fragmentada | Pipeline unificado | ✅ Robustez |
| **Configuration** | Limitada | Completamente configurable | ✅ Flexibilidad |
| **Safety** | Básica | Multi-nivel extrema | ✅ Protección total |

### 🎯 **VALIDACIÓN COMPLETA DEL SISTEMA**

```bash
# Tests de todas las nuevas funcionalidades
python test_improvements.py
# ✅ ALL 10 IMPROVEMENTS SUCCESSFULLY IMPLEMENTED AND TESTED

# Validación integrada end-to-end
python main.py --validate-improvements
# ✅ SYSTEM OPERATIONAL WITH ALL ENHANCEMENTS

# Performance metrics
python test_weight_calculator.py
# ✅ Weight calculator with correlation-based sizing: PASSED
```

### 📈 **BENEFICIOS CLAVE DEL SISTEMA HRM 2025**

1. **🚀 Rendimiento Superior**: Posiciones más grandes para señales de calidad
2. **🛡️ Riesgo Controlado**: Stop-loss dinámicos y profit-taking escalonado
3. **🔄 Adaptabilidad**: Sincronización BTC/ETH y rebalanceo automático
4. **⚡ Eficiencia**: Pipeline optimizado con configuración dinámica
5. **🔧 Robustez**: 10 capas de validación y controles de seguridad
6. **📊 Transparencia**: Logging completo y monitoreo en tiempo real

**El sistema HRM ahora incluye las 10 mejoras críticas completamente integradas y operativas, proporcionando un sistema de trading de nivel institucional con controles avanzados de riesgo y optimización inteligente de capital.**

### ✅ **COMPONENTES ACTUALIZADOS EN 2025**

#### 🎯 **19. Enhanced L3 Decision Maker**
- **Funcionalidad**: Toma de decisiones estratégica con lógica aware de setups de mercado
- **Setup-Aware Allocations**: Detecta setups OVERSOLD/OVERBOUGHT y ajusta allocations automáticamente
- **Regime-Specific Logic**: Lógica de decisión específica por régimen de mercado
- **Estado**: ✅ **OPERATIVO** - implementado en `l3_strategy/decision_maker.py`

#### 🎯 **20. Advanced L3 Regime Classifier**
- **Funcionalidad**: Classifier avanzado con detección de setups intrarégimen
- **Setup Detection**: OVERSOLD_SETUP y OVERBOUGHT_SETUP dentro de regímenes RANGE
- **Dynamic Windows**: Ventanas temporales dinámicas para análisis de 6 horas
- **Intelligent Thresholds**: RSI <40 (oversold), RSI >60 (overbought), ADX >25
- **Estado**: ✅ **OPERATIVO** - implementado en `l3_strategy/regime_classifier.py`

#### 🎯 **21. Complete L3 Technical Indicators Suite**
- **Funcionalidad**: Suite completa de indicadores técnicos para análisis de régimen
- **Advanced Indicators**: RSI, MACD, ADX, ATR, Bollinger Bands, Momentum, SMA/EMA
- **Data Validation**: Validación automática con fallback seguros
- **Scalability**: Optimizado para análisis multi-timeframe y alta frecuencia
- **Estado**: ✅ **OPERATIVO** - implementado en `l3_strategy/regime_features.py`

**¡Sistema HRM con L3 completamente mejorado y operativo!** 🎯⚡

---

## 🚀 **OPTIMIZACIONES RECIENTES 2025** - SISTEMA HRM MEJORADO

### ✅ **OPTIMIZACIONES IMPLEMENTADAS**

#### 🎯 **1. Optimización del Pipeline de Señales**
- **Reducción de señales HOLD**: Ajuste del sistema de votación L1+L2+L3 de 2/3 a 1/2 (50%) para mayor agilidad
- **Mejora de precisión**: Umbrales de confianza aumentados (0.3/0.2) para señales de mayor calidad
- **Filtrado inteligente**: Solo señales con alta confianza pasan a ejecución

#### 🔄 **2. Rebalanceo Automático de Portfolio**
- **Capital utilization óptima**: Rebalanceo automático cada 5 ciclos cuando hay >$500 disponibles
- **Asignación equal-weight**: Distribución automática entre símbolos activos
- **Conservative approach**: Máximo 30% del capital disponible por rebalanceo
- **Minimum order size**: Solo órdenes >$10 para evitar slippage

#### ⚡ **3. Eficiencia de Ciclo Mejorada**
- **Ciclo reducido**: De 10 a 8 segundos para mayor responsiveness
- **Procesamiento optimizado**: Menor latencia en generación de señales
- **Mejor sincronización**: Respuesta más rápida a cambios de mercado

#### 🏊 **4. Gestión Avanzada de Liquidez**
- **Validación de mercado**: Chequeo de volumen disponible antes de ejecutar órdenes
- **Prevención de slippage**: Máximo 5% del volumen promedio diario (10% en mercados altamente líquidos)
- **Análisis de volumen**: 20 períodos de volumen para evaluación precisa
- **Rechazo automático**: Órdenes que excedan límites de liquidez son rechazadas

#### 📊 **5. Validación de Datos Mejorada**
- **Más datos históricos**: Aumento de 50 a 200 puntos OHLCV para mejor análisis
- **Contexto técnico superior**: Más datos para indicadores y patrones
- **Señales más precisas**: Análisis basado en datos más completos

#### 🎛️ **6. Umbrales de Confianza Optimizados**
- **Confianza mínima**: 0.3 para señales base, 0.2 para fuerza
- **Filtrado de ruido**: Eliminación de señales de baja calidad
- **Mejor signal-to-noise ratio**: Solo señales con alto potencial pasan

### 📊 **IMPACTO ESPERADO DE LAS OPTIMIZACIONES**

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Señales HOLD** | Alto % | Reducido 50% | ✅ Mayor agilidad |
| **Utilización Capital** | Subóptima | Automática | ✅ Mejor eficiencia |
| **Tiempo de Ciclo** | 10s | 8s | ✅ +20% velocidad |
| **Slippage** | Potencial alto | Controlado | ✅ Riesgo reducido |
| **Calidad Señales** | Variable | Alta confianza | ✅ Mejor precisión |
| **Datos Análisis** | 50 puntos | 200 puntos | ✅ Mejor contexto |

### 🧪 **VALIDACIÓN DE OPTIMIZACIONES**
```bash
# Ejecutar tests de validación
python test_improvements.py

# Resultado esperado:
# ✅ ALL TESTS PASSED!
# ✅ Three solutions successfully implemented:
#    1. Validación Mejorada de Órdenes
#    2. Gestión Mejorada del Capital
#    3. Configuración Recomendada
```

### 🔧 **CONFIGURACIÓN DE OPTIMIZACIONES**

```python
# Parámetros optimizados en config
TRADING_CONFIG = {
    'MIN_ORDER_SIZE_USD': 5.0,          # Reducido para más señales
    'MAX_ALLOCATION_PER_SYMBOL_PCT': 30.0,  # Límite por símbolo
    'AVAILABLE_TRADING_CAPITAL_PCT': 80.0,  # 80% del capital disponible
    'CASH_RESERVE_PCT': 20.0,              # Reserva de seguridad
    'VALIDATION': {
        'ENABLE_ORDER_SIZE_CHECK': True,
        'ENABLE_CAPITAL_CHECK': True,
        'ENABLE_LIQUIDITY_CHECK': True,     # NUEVO: Chequeo de liquidez
        'ENABLE_POSITION_CHECK': True
    }
}
```

### 📈 **MONITOREO DE OPTIMIZACIONES**

**Logs mejorados para tracking:**
```
🔄 PORTFOLIO REBALANCING: Available capital $750 > $500 threshold
🔄 REBALANCING ORDER: BUY 0.0045 BTC @ $45000 (target: $250)
✅ Portfolio rebalancing completed: 2 orders executed
🏊 Liquidity check for BTCUSDT: order=$225, max_allowed=$1000, sufficient=true
⚡ Cycle 150 completed in 7.8s (optimized from 9.2s)
```

### 🎯 **BENEFICIOS CLAVE**

1. **🚀 Mayor Velocidad**: Ciclos 20% más rápidos
2. **💰 Mejor Capital Usage**: Rebalanceo automático inteligente
3. **🛡️ Menos Riesgo**: Validación de liquidez previene slippage
4. **🎯 Más Precisión**: Señales de mayor calidad
5. **📊 Mejor Análisis**: Más datos históricos para decisiones
6. **🔄 Mayor Agilidad**: Menos señales HOLD, más acción

**El sistema HRM ahora opera con optimizaciones de nivel institucional, maximizando eficiencia mientras mantiene controles de riesgo robustos.**

## 🎯 **OBJETIVO DEL PROYECTO**

Tomar decisiones de trading razonadas y trazables para múltiples activos (BTC, ETH) mediante una jerarquía de agentes. Aprender qué señales funcionan bajo distintos regímenes y cómo combinar niveles (L2/L3) para optimizar ejecución en L1 con modelos IA. Minimizar riesgos con análisis multinivel, capa dura de seguridad en L1 y gestión de correlación BTC–ETH. Crear un framework reutilizable para distintos universos de activos líquidos.

### 📚 **¿Qué queremos aprender a nivel de sistema?**
- Si el razonamiento multietapa mejora la estabilidad frente a un agente monolítico
- Qué señales funcionan en cada régimen y cómo combinarlas en L2/L3
- Cómo distribuir capital/ponderaciones entre modelos/estrategias

### 🎯 **Beneficios esperados**
- Mayor precisión mediante composición multiasset y modelos IA (LogReg, RF, LightGBM)
- Reducción de riesgo vía diversificación temporal, límite rígido en L1 y gestión de correlación BTC–ETH
- Adaptabilidad automática a distintos regímenes de mercado
- Razonamiento multi-variable con métricas granulares por activo (latencia, slippage, tasa de éxito)

### ⚙️ **Flujo general (visión de tiempos)**
- **Nivel 3:** Análisis Estratégico — horas
- **Nivel 2:** Táctica de Ejecución — minutos
- **Nivel 1:** Ejecución + Gestión de Riesgo — segundos
## 🏗️ ARQUITECTURA REAL DEL SISTEMA

### 🎯 **NIVEL 2 - TÁCTICO (L2)** ✅ IMPLEMENTADO Y MODULARIZADO
**Rol:** Generación inteligente de señales de trading
**Funciones operativas:**
- ✅ **Análisis técnico multi-timeframe** (RSI, MACD, Bollinger Bands)
- ✅ **Modelos FinRL pre-entrenados** con ensemble de predicciones
- ✅ **Composición de señales** con pesos dinámicos
- ✅ **Position sizing** con Kelly Criterion y vol-targeting
- ✅ **Controles de riesgo pre-ejecución** (stops, correlación, drawdown)
- ✅ **Stop-Loss y Take-Profit dinámicos** basados en volatilidad y confianza
- ✅ **Cálculo automático de SL/TP** por señal generada

#### 🏗️ **NUEVA ARQUITECTURA MODULAR L2 (2025)**
**Refactorización completa:** El monolítico `finrl_integration.py` ha sido dividido en módulos especializados:

```
l2_tactic/
├── 📄 __init__.py                    # Punto de entrada unificado
├── 📄 models.py                      # Estructuras de datos (TacticalSignal)
├── 📄 config.py                      # Configuración L2
├── 📄 tactical_signal_processor.py    # Orquestador principal
├── 📄 signal_composer.py             # Composición de señales
├── 📄 position_sizer.py              # Cálculo de tamaños de posición
├── 📄 finrl_integration.py           # 🔄 COMPATIBILIDAD (solo imports)
├── 📁 generators/                    # Generadores de señales
│   ├── 📄 __init__.py
│   ├── 📄 technical_analyzer.py      # Análisis técnico
│   ├── 📄 mean_reversion.py          # Estrategia reversión a la media
│   └── 📄 finrl.py                   # Procesador FinRL
├── 📁 ensemble/                      # Combinación de señales
│   ├── 📄 __init__.py
│   ├── 📄 voting.py                  # Ensemble por votación
│   └── 📄 blender.py                 # Ensemble por blending
├── 📁 risk_controls/                 # Gestión de riesgo modular
│   ├── 📄 __init__.py
│   ├── 📄 alerts.py                  # Sistema de alertas
│   ├── 📄 manager.py                 # Orquestador de riesgo
│   ├── 📄 portfolio.py               # Riesgo de portfolio
│   ├── 📄 positions.py               # Riesgo por posición
│   └── 📄 stop_losses.py             # Stop-loss dinámicos
└── 📁 technical/                     # Indicadores técnicos
    ├── 📄 __init__.py
    ├── 📄 multi_timeframe.py         # Análisis multi-timeframe
    └── 📄 indicators.py              # Indicadores técnicos
```

#### 🤖 **Sistema FinRL Modularizado**
**Antes:** Un solo archivo de 1000+ líneas con todo mezclado
**Ahora:** Arquitectura limpia con responsabilidades separadas:

| Módulo | Responsabilidad | Estado |
|--------|----------------|--------|
| `finrl_processor.py` | Clase principal FinRLProcessor | ✅ Operativo |
| `finrl_wrapper.py` | Wrapper inteligente multi-modelo | ✅ Operativo |
| `feature_extractors.py` | Extractores de features personalizados | ✅ Operativo |
| `observation_builders.py` | Construcción de observaciones | ✅ Operativo |
| `model_loaders.py` | Carga unificada de modelos | ✅ Operativo |
| `signal_generators.py` | Generación de señales | ✅ Operativo |

#### 🎯 **Modelos FinRL Soportados**
| Modelo | Dimensiones | Método | Estado |
|--------|-------------|--------|--------|
| **DeepSeek** | 257 | `predict()` | ✅ Operativo |
| **Gemini** | 13 | `get_action()` → `predict()` | ✅ **FIXED** |
| **Claude** | 971 | `predict()` | ✅ Operativo |
| **Kimi** | Variable | `predict()` | ✅ Operativo |
| **Gpt** | Variable | `predict()` | ✅ Operativo |
| **Grok** | Variable | `predict()` | ✅ Operativo |

#### 🔧 **Detección Automática de Métodos**
```python
# Sistema inteligente que detecta el método correcto
def get_finrl_signal(finrl_processor, market_data):
    if hasattr(finrl_processor, 'predict'):
        return finrl_processor.predict(market_data)
    elif hasattr(finrl_processor, 'get_action'):
        return finrl_processor.get_action(market_data)
    else:
        raise AttributeError("Método no encontrado")
```

#### 📈 **Beneficios de la Modularización**
- **🔧 Mantenibilidad:** Cada módulo tiene una responsabilidad clara
- **🔄 Escalabilidad:** Fácil añadir nuevos modelos o estrategias
- **🛡️ Robustez:** Mejor manejo de errores y compatibilidad
- **📊 Rendimiento:** Optimizaciones específicas por componente
- **🔌 Compatibilidad:** Código existente sigue funcionando sin cambios

### ⚙️ **NIVEL 1 - OPERACIONAL (L1)** ✅ IMPLEMENTADO
**Rol:** Ejecución determinista y segura de órdenes
**Funciones operativas:**
- ✅ **Validación de señales** con modelos IA (LogReg, RF, LightGBM)
- ✅ **Gestión de portfolio automática** (BTC, ETH, USDT)
- ✅ **Conexión a Binance Spot** (real y testnet)
- ✅ **Logging persistente** con métricas detalladas
- ✅ **Controles de riesgo** por símbolo y portfolio

### 🚀 **NIVEL L3** - ESTRATÉGICO (IMPLEMENTADO)
**Rol:** Análisis macro y asignación estratégica de capital
**Funciones implementadas:**
- ✅ **Regime Detection** con ensemble ML (Optuna)
- ✅ **Portfolio Optimization** usando Black-Litterman
- ✅ **Sentiment Analysis** con BERT pre-entrenado (Reddit + News API)
- ✅ **Volatility Forecasting** con GARCH y LSTM
- ✅ **Strategic Decision Making** con pipeline completo

✅ **Modelos IA L1:** **FUNCIONALES** (LogReg, RF, LightGBM en models/L1/)

| Tipo | Descripción |
|------|-------------|
| **Precio** | delta_close, EMA/SMA |
| **Volumen** | volumen relativo |
| **Momentum** | RSI, MACD |
| **Multi-timeframe** | 1m + 5m |
| **Cross-asset** | ETH/BTC ratio, correlación rolling, divergencias |
| **Real-time data** | Desde Binance Spot (modo LIVE) o testnet |
## 🚀 EJECUCIÓN DEL SISTEMA

### ⚡ **INICIO RÁPIDO**
```bash
# 1) Configurar variables de entorno (opcional para modo simulado)
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_secret_key
export USE_TESTNET=true  # false para modo LIVE

# 2) Ejecutar sistema principal (modo simulado por defecto)
python main.py

# 3) Para ejecución nocturna continua
python run_overnight.py
```

### 🎯 **MODO SIMULADO CON 3000 USDT (RECOMENDADO PARA TESTING)**
```bash
# Sin configuración adicional - funciona inmediatamente
python main.py

# El sistema inicia con:
# - Balance inicial: 3000.0 USDT
# - Portfolio completamente limpio
# - Sin conexión a exchanges reales
# - Todas las funcionalidades activas
```

### 📊 **FUNCIONAMIENTO EN TIEMPO REAL**
El sistema ejecuta un **ciclo principal cada 10 segundos**:

1. **📈 Recolección de datos:** Obtiene OHLCV de Binance para BTC/ETH
2. **🧮 Cálculo de indicadores:** RSI, MACD, Bollinger Bands, volatilidad
3. **🤖 Procesamiento L2:** Genera señales con modelos FinRL + análisis técnico
4. **🛡️ Cálculo SL/TP:** Stop-loss y take-profit dinámicos por señal
5. **⚙️ Procesamiento L1:** Valida señales y ejecuta órdenes seguras
6. **💰 Actualización portfolio:** Tracking automático con costos reales
7. **🔍 Monitoreo posiciones:** Activación automática de SL/TP
8. **📝 Logging persistente:** Guarda métricas en data/logs/ y data/portfolio/

### 🔄 **ANÁLISIS DE SENTIMIENTO EN TIEMPO REAL**
**Cada 50 ciclos (~8-9 minutos):**
```
🔄 SENTIMENT: Actualización periódica iniciada (ciclo 50, cada 50 ciclos)
🔄 SENTIMENT: Iniciando actualización de datos de sentimiento...
📱 SENTIMENT: Descargando datos de Reddit...
📱 SENTIMENT: r/CryptoCurrency - Descargados 500 posts
📱 SENTIMENT: r/Bitcoin - Descargados 500 posts
📱 SENTIMENT: r/Ethereum - Descargados 500 posts
📊 SENTIMENT: Reddit total descargado: 1500 posts de 3 subreddits
📰 SENTIMENT: News - 50 artículos descargados y procesados
💬 SENTIMENT: Análisis de sentimiento listo con 95 textos válidos
💬 SENTIMENT: Cache actualizado con 95 textos para análisis L3
```

**Cada ciclo L3:**
```
🧠 SENTIMENT: Iniciando inferencia de sentimiento - 95 textos, batch_size=16
📊 SENTIMENT: Procesando 6 batches de inferencia...
✅ SENTIMENT: Completado batch 6/6 (100.0%)
🎯 SENTIMENT: Inferencia completada - 95 resultados generados
✅ Sentimiento calculado: 0.2345 (device: cpu, textos: 95)
🟠 ANÁLISIS DE SENTIMIENTO: 🟠 POSITIVO - Mercado favorable, tendencia alcista moderada (score: 0.2345)
```

### ⏰ **FRECUENCIAS DE EJECUCIÓN**
- **L2/L1:** Cada 10 segundos (independiente)
- **L3:** Cada 50 ciclos (~8-9 minutos) en segundo plano
- **Sentiment Analysis:** Cada 50 ciclos (descarga fresca de datos)
- **Si L3 falla:** L2 usa última estrategia conocida (fallback automático)

### **VENTAJAS DEL FALLBACK**
- L2/L1 nunca se bloquea si L3 falla
- Última estrategia válida de L3 se mantiene
- Logs centralizados registran errores y warnings
- Sentiment analysis continúa con datos en cache

### 🎛️ **MODOS DE OPERACIÓN**
| Modo | Descripción | Activación | Balance Inicial |
|------|-------------|------------|----------------|
| **SIMULATED** | Simulación completa sin exchange | Automático | 3000 USDT |
| **TESTNET** | Binance testnet | `USE_TESTNET=true` | Desde exchange |
| **LIVE** | Binance Spot real | `USE_TESTNET=false` | Desde exchange |
| **PAPER** | Simulación local | Configuración interna | Configurable |

## 🤖 **SISTEMA DE AUTO-APRENDIZAJE CON PROTECCIÓN ANTI-OVERFITTING**

**NUEVA FUNCIONALIDAD 2025:** HRM ahora incluye un **sistema de aprendizaje continuo completamente automático** con **9 capas de protección anti-overfitting**. El sistema aprende y se mejora solo sin intervención manual.

### 🎯 **Características del Sistema de Auto-Aprendizaje**

#### ✅ **Aprendizaje Continuo Automático**
- **Reentrenamiento automático** basado en triggers inteligentes
- **Online learning** para componentes compatibles
- **Meta-learning** para selección automática de modelos
- **Ensemble evolution** dinámica

#### 🛡️ **Protección Total Anti-Overfitting (9 Capas)**

1. **🔄 Validación Cruzada Continua** - Rolling window validation
2. **📊 Regularización Adaptativa** - Ajuste automático de parámetros
3. **🧬 Ensemble Diverso** - Modelos diversos para estabilidad
4. **⏹️ Early Stopping Inteligente** - Prevención de sobre-entrenamiento
5. **🌊 Concept Drift Detection** - Detección de cambios en distribución
6. **📈 Walk-Forward Validation** - Validación temporal realista
7. **🎨 Data Diversity Enforcement** - Garantía de diversidad en datos
8. **🏷️ Model Aging Detection** - Detección de degradación de modelos
9. **📉 Out-of-Sample Testing** - Validación en datos no vistos

### 🚀 **Arquitectura del Sistema de Auto-Aprendizaje**

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA PRINCIPAL                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Auto-Retraining System                       │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │        Anti-Overfit Protection (9 capas)           │ │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │ │ │
│  │  │  │      Model Validation & Selection              │ │ │ │
│  │  │  └─────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Online Learning Components                  │ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Performance Monitor                         │ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 📊 **Triggers de Auto-Reentrenamiento**

#### ⏰ **Basado en Tiempo**
- **Cada 7 días** automáticamente
- Reset automático de timers

#### 📈 **Basado en Performance**
- **Win rate < 52%** en últimos 100 trades
- **Drawdown > 12%** máximo
- **Auto-detección** de degradación

#### 🔄 **Basado en Régimen**
- **3 cambios de régimen** consecutivos
- **Adaptación automática** a nuevos mercados

#### 📊 **Basado en Volumen**
- **500+ nuevos trades** acumulados
- **Datos suficientes** para reentrenamiento significativo

### 🔧 **Componentes Implementados**

#### 1. **`auto_learning_system.py`** - 🧠 Sistema Principal
- **9 clases principales** con protección anti-overfitting
- **Auto-reentrenamiento automático** con triggers inteligentes
- **Validación cruzada continua**, **regularización adaptativa**, **ensemble diverso**
- **Concept drift detection**, **early stopping inteligente**
- **Sistema completamente autónomo**

#### 2. **`integration_auto_learning.py`** - 🔗 Integración
- **Conexión automática** con el sistema de trading principal
- **Parsing automático** de logs para capturar trades
- **Hook de logging** para aprendizaje en tiempo real
- **Función de integración** plug-and-play

#### 3. **`README_AUTO_LEARNING.md`** - 📖 Documentación Completa
- **Documentación detallada** del sistema
- **Guía de integración** paso a paso
- **Arquitectura detallada** y funcionalidades
- **Monitoreo y métricas**

### 🎯 **Funcionalidades del Sistema**

#### ✅ **Auto-Reentrenamiento**
```python
# El sistema decide automáticamente cuándo reentrenar
if self._should_retrain():
    await self._auto_retrain_models()
```

#### ✅ **Protección Anti-Overfitting**
```python
# TODAS las verificaciones pasan antes de desplegar modelo
if self._passes_all_anti_overfitting_checks(candidate_model, training_data):
    self._deploy_new_model(model_name, candidate_model)
```

#### ✅ **Ensemble Evolution**
```python
# Solo añade modelos que aumenten diversidad
if self.ensemble_builder.add_model_to_ensemble(candidate_model, validation_data):
    logger.info("✅ Model added to ensemble")
```

#### ✅ **Concept Drift Detection**
```python
# Detecta cambios en la distribución de datos
if self.drift_detector.detect_drift(new_data):
    logger.warning("🌊 CONCEPT DRIFT DETECTED")
```

### 📈 **Beneficios Esperados**

#### 🚀 **Mejora Continua**
- **Win rate**: 55% → 65%+ en 3-6 meses
- **Drawdown máximo**: 15% → 10%+
- **Adaptabilidad**: Auto-ajuste a cambios de mercado

#### 🛡️ **Riesgo Controlado**
- **Sin overfitting**: 9 capas de protección
- **Validación robusta**: Múltiples técnicas
- **Stability**: Ensemble diverso

#### 🤖 **Autonomía Total**
- **Sin intervención**: Funciona 24/7
- **Auto-optimización**: Parámetros ajustados automáticamente
- **Auto-evolución**: Modelos mejoran solos

### 🔌 **Integración Automática**

**El sistema de auto-aprendizaje se integra automáticamente al iniciar HRM:**

```python
# En main.py - integración automática
from integration_auto_learning import integrate_with_main_system

# Integrar al inicio
auto_learning_system = integrate_with_main_system()
```

### 📊 **Monitoreo del Sistema de Auto-Aprendizaje**

#### **Estado del Sistema**
```python
{
    'integrated_system_running': True,
    'auto_learning_active': True,
    'trades_processed': 1250,
    'auto_learning_status': {
        'data_buffer_size': 500,
        'models_count': 5,
        'ensemble_size': 3,
        'performance_metrics': {...}
    }
}
```

#### **Logs Automáticos**
```
🔄 AUTO-TRIGGER: Time-based (192h >= 168h)
🤖 INICIANDO AUTO-REENTRENAMIENTO...
✅ CV Validation passed: 0.73 ± 0.08
✅ Model added to ensemble (improvement: 0.023)
🚀 Desplegado regime_classifier versión auto_v3
```

### 🎉 **Resultado Final**

**Sistema HRM con aprendizaje continuo automático:**
- ✅ **Se mejora solo** sin intervención manual
- ✅ **Aprende de cada trade** automáticamente
- ✅ **Previene overfitting** con 9 capas de protección
- ✅ **Se adapta** a cambios de mercado
- ✅ **Funciona 24/7** de forma autónoma

**¡HRM ahora tiene aprendizaje continuo con protección total anti-overfitting!** 🤖🛡️✨

---

## ✅ **BUENAS PRÁCTICAS DE RIESGO** (resumen actualizado)

| Concepto | Valor real |
|----------|------------|
| **Stop-loss** | Obligatorio + automático |
| **Take-profit** | Dinámico basado en volatilidad |
| **Límites por trade** | BTC: 0.05, ETH: 1.0 |
| **Exposición máxima** | BTC: 20%, ETH: 15% |
| **Correlación BTC-ETH** | Monitoreada en tiempo real |
| **Costos reales** | Comisiones 0.1% Binance aplicadas |
| **Monitoreo posiciones** | Activación automática SL/TP |
| **Modo LIVE** | Implementado y validado |
| **Auto-aprendizaje** | ✅ **NUEVO** - Sistema autónomo con 9 capas anti-overfitting |
| **Determinismo** | Una orden por señal → si falla → rechazo y reporte |
| **Separación L2/L3 ≠ L1** | Responsabilidades claramente separadas |

## 🏗️ Arquitectura del Sistema HRM

### 📊 Flujo Jerárquico de Decisiones

```
🌐 NIVEL 3 (ESTRATÉGICO) - Análisis Macro (cada 10 min)
├── 📊 Análisis de Mercado (Regime Detection)
├── 💬 Análisis de Sentimiento (BERT + Redes Sociales)
├── 📈 Pronóstico de Volatilidad (GARCH + LSTM)
└── 🎯 Optimización de Portfolio (Black-Litterman)
    ↓
🎯 NIVEL 2 (TÁCTICO) - Generación de Señales (cada 10 seg)
├── 🤖 Modelos FinRL (DeepSeek, Gemini, Claude, Kimi)
├── 📊 Análisis Técnico Multi-Timeframe
├── 🎲 Ensemble de Señales con Ponderación Dinámica
└── 🛡️ Controles de Riesgo Pre-Ejecución
    ↓
⚡ NIVEL 1 (OPERACIONAL) - Ejecución Determinista
├── 🔍 Validación de Señales con Modelos IA
├── 💰 Gestión Automática de Portfolio
├── 🔗 Conexión Binance (Live/Testnet)
└── 📝 Logging Persistente y Métricas
```



🔗 6️⃣ Conexión entre niveles (resumen actualizado)

Flujo	Descripción
L3 → L2	Selección de sub-estrategias y universo (BTC, ETH)
L2 → L1	Señales concretas (cantidad, stop, target) por símbolo
L1 → Exchange	Envío/gestión de órdenes en tiempo real para BTC/USDT y ETH/USDT desde Binance Spot o testnet

### MÓDULOS CORE ✅ IMPLEMENTADOS
Funcionalidades esenciales:
core/state_manager.py - Gestión del estado del sistema
core/portfolio_manager.py - Tracking y gestión de portfolio
core/technical_indicators.py - Cálculo de indicadores
core/feature_engineering.py - Preparación de features para L2
🔹 Logging centralizado:
Todos los módulos usan un único logger centralizado en core/logging.py, que combina:
  - Logging estándar de Python.
  - Loguru para formatos enriquecidos y colores en consola.
  - Trazabilidad de ciclo, símbolo y nivel.

## 📂 7️⃣ Estructura de carpetas

```text
HRM/
│── docs/                      
│
│── storage/                   
│   ├── csv_writer.py
│   ├── sqlite_writer.py
│   └── __init__.py
│
├── core/     
│   ├── __init__.py
│   ├── state_manager.py         # Gestión del estado global
│   ├── portfolio_manager.py     # Gestión de portfolio y CSV
│   ├── technical_indicators.py  # Cálculo de indicadores técnicos
│   ├── feature_engineering.py   # Preparación de features para L2          
│   ├── logging.py
│   ├── scheduler.py
│   └── utils.py
│
├── comms/                     
│   ├── config/                
│   ├── message_bus.py
│   ├── schemas.py
│   └── adapters/
│
├── l3_strategy/              
│   ├── __init__.py
│   ├── README.md  
│   ├── models.py
│   ├── config.py
│   ├── strategic_processor.py
│   ├── bus_integration.py
│   ├── performance_tracker.py
│   ├── metrics.py
│   ├── procesar_l3.py
│   ├── ai_model_loader.py
│   └── ai_models/
│       ├── unified_decision_model.py
│       ├── regime_detector.py
│       └── risk_assessor.py
│
├── l2_tactic/                 
│   ├── signal_generator.py
│   ├── position_sizer.py
│   ├── risk_controls.py
│   └── __init__.py
│
├── l1_operational/            
│   ├── models.py
│   ├── config.py
│   ├── bus_adapter.py
│   ├── order_manager.py
│   ├── risk_guard.py
│   ├── executor.py
│   ├── data_feed.py
│   ├── binance_client.py
│   ├── ai_models/
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── test_clean_l1_multiasset.py
│   ├── README.md
│   └── requirements.txt
│
├── models/                    
│   ├── L1/
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── L2/
│   ├── L3/
│
├── data/                      
│   ├── connectors/
│   │   └── binance_connector.py
│   ├── loaders.py
│   ├── storage/
│   └── __init__.py
│
├── risk/                      
│   ├── limits.py
│   ├── var_es.py
│   ├── drawdown.py
│   └── __init__.py
│
├── monitoring/                
│   ├── dashboards/
│   ├── alerts.py
│   ├── telemetry.py
│   └── __init__.py
│
├── tests/                     
│   └── backtester.py
└── main.py
```

> **Nota:** Esta estructura resume el proyecto real y es suficiente para navegar y extender el código.

---

## 🔁 TABLA DE TIEMPOS/FRECUENCIAS
| Nivel | Frecuencia              |
| ----- | ----------------------- |
| L3    | 10 min (periódico)      |
| L2    | 10 s                    |
| L1    | subsegundos / inmediato |


## 🔁 8️⃣ Flujo de mensajes y state global

Cada ciclo trabaja sobre un único `state` (dict). Cada nivel actualiza su sección para trazabilidad y debugging.

```python
state = {
    "mercado": {...},       # precios actuales por símbolo (BTC, ETH)
    "estrategia": "...",    # estrategia activa (agresiva/defensiva)
    "portfolio": {...},     # asignación de capital
    "universo": [...],      # activos (BTC/USDT, ETH/USDT)
    "exposicion": {...},    # % exposición por activo
    "senales": {...},       # señales tácticas por símbolo
    "ordenes": [...],       # órdenes ejecutadas en L1
    "riesgo": {...},        # chequeo de riesgo (incluye correlación BTC-ETH)
    "deriva": False,        # drift detection
    "ciclo_id": 1
}
```

**Flujo L1 (ejecución determinista):**
L2/L3 → Bus Adapter → Order Manager → Hard-coded Safety → AI Models (LogReg, RF, LightGBM) → Risk Rules → Executor → Exchange → Execution Report → Bus Adapter → L2/L3

---

## ✅ 9️⃣ L1\_operational — “limpio y determinista”

**L1 NO hace**

* ❌ No modifica cantidades ni precios de señales estratégicas.
* ❌ No decide estrategia ni táctica.
* ❌ No actualiza portfolio completo (responsabilidad de L2/L3).
* ❌ No recolecta ni procesa datos de mercado (responsabilidad de L2/L3).

**L1 SÍ hace**

* ✅ Validar límites de riesgo por símbolo (stop-loss, exposición, correlación BTC-ETH).
* ✅ Filtrar señales con modelos IA para confirmar tendencias.
* ✅ Ejecutar órdenes pre-validadas (modo PAPER simulado).
* ✅ Generar reportes detallados por activo.
* ✅ Mantener trazabilidad completa con métricas por símbolo.

**Verificación de limpieza:**
`python l1_operational/test_clean_l1_multiasset.py`

---

## 🔌 Mensajería, logging y telemetría

* **Mensajería:** `comms/` define esquemas y bus (JSON/Protobuf). Colas asyncio; adapters Kafka/Redis opcionales.
* **Logging estructurado:** JSON (ej.: `python-json-logger`) con etiquetas por símbolo (`[BTC]`, `[ETH]`).
* **Telemetría (monitoring/telemetry.py):**

  * `incr(name)` → contadores (órdenes por símbolo)
  * `gauge(name, value)` → métricas instantáneas (exposición, correlación)
  * `timing(name, start)` → latencias por ejecución

**Dashboard consola:** Visualización con `rich` por ciclo (métricas por activo).

---

## 🗃️ Persistencia de histórico

Cada ciclo se guarda en:

* **CSV:** `data/historico.csv` (todas las variables del `state`).
* **SQLite:** `data/historico.db` (tabla `ciclos` con los mismos datos).

Permite exportar a pandas/Excel, reproducir backtests y consultar con SQL.

---

## 🧪 Dataset & features (BTC/USDT, ETH/USDT)

Generador de features en `data/loaders.py`. Soporta 1m + 5m (multi-timeframe). Índice `datetime` y columna `close`.

**Features incluidas**

* Precio: `delta_close`, `ema_10/20`, `sma_10/20`
* Volumen: `vol_rel` vs media N (20)
* Momentum: `rsi`, `macd`, `macd_signal`, `macd_hist`
* Multi-timeframe: 1m + 5m (`_5m`)
* Cruzadas: `ETH/BTC ratio`, correlación rolling, divergencias

**Uso básico**

```python
import pandas as pd
from data.loaders import prepare_features

# 1) Cargar velas 1m
df_btc_1m = pd.read_csv("data/btc_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df_eth_1m = pd.read_csv("data/eth_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# 2) Generar features 1m+5m y split temporal (80/20 por defecto)
train_btc, test_btc = prepare_features(df_btc_1m, test_size=0.2, symbol="BTC")
train_eth, test_eth = prepare_features(df_eth_1m, test_size=0.2, symbol="ETH")

# 3) Guardar datasets
train_btc.to_csv("data/btc_features_train.csv")
test_btc.to_csv("data/btc_features_test.csv")
train_eth.to_csv("data/eth_features_train.csv")
test_eth.to_csv("data/eth_features_test.csv")
```

> **Nota:** Si ya tienes velas 5m, pásalas como `df_5m` para evitar resampleo. Si tu CSV trae `BTC_close` o `ETH_close`, `normalize_columns` lo mapea a `close` automáticamente.

---

## ⚙️ Puesta en marcha

**Requisitos**

* Python 3.10+
* Cuenta de exchange (modo sandbox recomendado para L1)
* Credenciales/API Keys (env vars o `.env`)
* `pip`, `venv`

**Instalación rápida**

```bash
# 1) Clonar
git clone https://github.com/t3rm1nus/HRM.git
cd HRM

# 2) Entorno
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Dependencias (L1)
pip install -r l1_operational/requirements.txt

# 4) (Opcional) Dependencias extra según adapters
# pip install -r requirements.txt
```

**Ejecución (demo)**

```bash
python main.py
```

Configurar parámetros y límites en `core/config/` y en variables de entorno.

---

## ✅ Buenas prácticas de riesgo (resumen)

* **Hard limits en L1:** Stop-loss obligatorio.
* Límites por trade: BTC: `0.05` max, ETH: `1.0` max.
* Exposición máxima: BTC: `20%`, ETH: `15%`.
* Chequeos de liquidez/saldo, drawdown y correlación BTC-ETH.
* **Determinismo:** Una oportunidad de orden por señal; si no cumple reglas → rechazo y reporte.
* **Separación de responsabilidades:** Señal (L2/L3) ≠ Ejecución (L1).
* **Backtesting:** Histórico persistido + state reproducible.

---

## 🧩 Tests e integración

* Pruebas de limpieza L1: `l1_operational/test_clean_l1_multiasset.py`
* Backtester E2E: `tests/backtester.py`
* Métricas/alertas: `monitoring/` (métricas por símbolo y correlación)

---

## 🛣️ Roadmap (alto nivel)

* Mejores clasificadores de régimen (L3)
* Ensamble multi-señal robusto (L2)
* Integración multi-exchange/DEX y simulador de slippage (L1)
* Dashboards web y alertas proactivas con métricas por activo

---

## 👥 Autoría y licencia

**Autoría:** Equipo de desarrollo HRM
**Versión:** 1.0
**Última actualización:** 2025
**Licencia:** Ver `LICENSE` si aplica

---

## 📝 Notas finales

Este README está diseñado para ser **autosuficiente**: describe la jerarquía, arquitectura, flujos, estructura de código, dataset, telemetría, persistencia y puesta en marcha para que un agente externo/colaborador comprenda y opere el proyecto sin necesidad inmediata de otros documentos.
