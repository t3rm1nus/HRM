# 📊 Dashboard HRM - Panel de Control

Dashboard web para monitorear el Sistema HRM (Hierarchical Reasoning Model) de Trading Algorítmico.

## 🚀 Instalación y Ejecución

### Prerrequisitos
- Node.js 16+ 
- npm o yarn

### Pasos para ejecutar:

1. **Instalar dependencias:**
```bash
npm install
```

2. **Ejecutar en modo desarrollo:**
```bash
npm start
```

3. **Abrir en el navegador:**
El dashboard se abrirá automáticamente en `http://localhost:3000`

## 📁 Estructura del Proyecto

```
├── public/
│   └── index.html          # HTML principal
├── src/
│   ├── components/
│   │   └── CryptoPortfolioDashboard.jsx  # Dashboard principal
│   ├── App.js              # Componente raíz
│   └── index.js            # Punto de entrada
├── package.json            # Dependencias
└── README_DASHBOARD.md     # Este archivo
```

## 🎯 Funcionalidades

### 4 Pestañas Principales:

1. **📈 Cartera**
   - Valor total del portfolio
   - Precios BTC/ETH en tiempo real
   - Evolución de la cartera (gráfico área)
   - Composición por activo (gráfico circular)
   - Evolución de precios (gráfico líneas)

2. **⚡ L1 Operacional**
   - Latencia del sistema
   - Throughput
   - Success rates por activo
   - Métricas detalladas BTC/ETH
   - Correlación BTC-ETH

3. **🧠 Modelos IA**
   - Performance de los 3 modelos:
     - Logistic Regression
     - Random Forest
     - LightGBM
   - Métricas: Accuracy, F1 Score, AUC
   - Flujo de decisión jerárquico

4. **🛡️ Gestión de Riesgo**
   - Límites de exposición por activo
   - Validaciones de seguridad
   - Matriz de límites de riesgo
   - Sistema de alertas

## 🔧 Tecnologías Utilizadas

- **React 18** - Framework de UI
- **Recharts** - Gráficos interactivos
- **Lucide React** - Iconos
- **Tailwind CSS** - Estilos (via CDN)

## 📊 Datos de Ejemplo

El dashboard incluye datos de ejemplo basados en el sistema HRM real:
- Portfolio con BTC, ETH, USDT
- Métricas L1 operacionales
- Performance de modelos IA
- Límites de riesgo reales

## 🌐 Acceso Web

Una vez ejecutado, el dashboard estará disponible en:
- **URL:** `http://localhost:3000`
- **Puerto:** 3000 (configurable)
- **Modo:** Desarrollo con hot reload

## 🔄 Actualizaciones en Tiempo Real

- Los datos se cargan automáticamente al iniciar
- Interfaz responsive para móvil y desktop
- Gráficos interactivos con tooltips
- Estados de carga y error

## 📱 Responsive Design

El dashboard está optimizado para:
- **Desktop:** Pantallas grandes con layout completo
- **Tablet:** Layout adaptativo
- **Móvil:** Navegación optimizada

## 🚨 Notas Importantes

- **Datos de ejemplo:** El dashboard usa datos simulados del sistema HRM
- **Conexión real:** Para datos reales, configurar conexión a `data/portfolio/`
- **Modo PAPER:** El sistema opera en modo simulación por seguridad

## 🔧 Personalización

Para conectar con datos reales del sistema HRM:
1. Modificar `loadPortfolioData()` en `CryptoPortfolioDashboard.jsx`
2. Conectar con archivos CSV de `data/portfolio/`
3. Implementar WebSocket para actualizaciones en tiempo real

---

**¡El dashboard está listo para usar!** 🎉
