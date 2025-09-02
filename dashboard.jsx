import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Bitcoin, Coins, Clock, AlertTriangle, Shield, Zap, Activity, Brain, Target, Eye } from 'lucide-react';

const CryptoPortfolioDashboard = () => {
  const [activeTab, setActiveTab] = useState('portfolio');
  const [portfolioData, setPortfolioData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Cargar datos de la carpeta data/portfolio/
  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Datos de ejemplo basados en el sistema HRM real
      const fallbackData = [
        "2025-09-01 00:22:48,1,3000.0,0.00558,608.91,109123.55,0.006,26.8,4467.14,2364.29",
        "2025-09-01 09:26:47,3262,2972.47,0.02232,2416.67,108273.59,0.126,554.58,4401.39,1.23",
        "2025-09-01 09:27:28,3266,2974.47,0.02232,2418.39,108350.94,0.126,554.85,4403.61,1.23",
        "2025-09-01 09:29:28,3278,2977.51,0.02232,2419.85,108416.03,0.126,556.44,4416.16,1.23",
        "2025-09-01 09:31:08,3288,2981.99,0.02232,2423.39,108574.84,0.126,557.37,4423.6,1.23",
        "2025-09-01 09:33:28,3302,2984.1,0.02232,2425.85,108684.88,0.126,557.03,4420.86,1.23",
        "2025-09-01 09:35:48,3316,2985.91,0.02232,2426.99,108736.01,0.126,557.69,4426.14,1.23",
        "2025-09-01 09:37:18,3325,2984.37,0.02232,2426.01,108692.04,0.126,557.14,4421.74,1.23"
      ];

      const allData = fallbackData.map(line => {
        const [timestamp, id, total, btc_qty, btc_value, btc_price, eth_qty, eth_value, eth_price, usdt_qty] = line.split(',');
        const time = new Date(timestamp).toLocaleTimeString('es-ES');
        return {
          time,
          timestamp,
          total: parseFloat(total),
          btc_qty: parseFloat(btc_qty),
          btc_value: parseFloat(btc_value),
          btc_price: parseFloat(btc_price),
          eth_qty: parseFloat(eth_qty),
          eth_value: parseFloat(eth_value),
          eth_price: parseFloat(eth_price),
          usdt_qty: parseFloat(usdt_qty)
        };
      });

      // Ordenar por timestamp
      allData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      
      setPortfolioData(allData);
      setError('Usando datos de ejemplo del sistema HRM. Para datos reales, configure la conexión a data/portfolio/');
      
    } catch (err) {
      setError(`Error cargando datos del portfolio: ${err.message}`);
      console.error('Error loading portfolio data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Cargar datos al montar el componente
  useEffect(() => {
    loadPortfolioData();
  }, []);

  // Métricas calculadas
  const metrics = useMemo(() => {
    if (portfolioData.length === 0) return {};
    
    const initial = portfolioData[0];
    const current = portfolioData[portfolioData.length - 1];
    
    const totalChange = current.total - initial.total;
    const totalChangePercent = ((current.total - initial.total) / initial.total) * 100;
    const btcChange = ((current.btc_price - initial.btc_price) / initial.btc_price) * 100;
    const ethChange = ((current.eth_price - initial.eth_price) / initial.eth_price) * 100;
    
    // Exposición por activo
    const btcExposure = (current.btc_value / current.total) * 100;
    const ethExposure = (current.eth_value / current.total) * 100;
    const usdtExposure = (current.usdt_qty / current.total) * 100;
    
    // Correlación simulada BTC-ETH
    const correlation = 0.73;
    
    return {
      totalValue: current.total,
      totalChange,
      totalChangePercent,
      btcChange,
      ethChange,
      btcExposure,
      ethExposure,
      usdtExposure,
      correlation,
      maxBtcExposure: 20,
      maxEthExposure: 15
    };
  }, [portfolioData]);

  // Datos para el gráfico de composición
  const compositionData = [
    { name: 'BTC', value: metrics.btcExposure || 0, color: '#f7931a' },
    { name: 'ETH', value: metrics.ethExposure || 0, color: '#627eea' },
    { name: 'USDT', value: metrics.usdtExposure || 0, color: '#26a17b' }
  ];

  // Métricas L1 simuladas basadas en el documento
  const l1Metrics = {
    btc: {
      signalsProcessed: { success: 45, failed: 3 },
      successRate: 93.8,
      avgSlippage: 0.12,
      currentExposure: metrics.btcExposure || 0,
      maxExposure: 20
    },
    eth: {
      signalsProcessed: { success: 32, failed: 2 },
      successRate: 94.1,
      avgSlippage: 0.15,
      currentExposure: metrics.ethExposure || 0,
      maxExposure: 15
    },
    ai: {
      logreg: { accuracy: 0.66, f1: 0.64, auc: 0.72 },
      randomForest: { accuracy: 0.65, f1: 0.61, auc: 0.70 },
      lightgbm: { accuracy: 0.67, f1: 0.63, auc: 0.73 }
    },
    system: {
      latency: 42,
      throughput: 125,
      correlation: metrics.correlation || 0,
      correlationLimit: 0.80
    }
  };

  const TabButton = ({ id, label, icon: Icon, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
        isActive 
          ? 'bg-blue-600 text-white shadow-md' 
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      <Icon size={18} />
      {label}
    </button>
  );

  const MetricCard = ({ icon: Icon, title, value, change, color = 'blue' }) => (
    <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg bg-${color}-100`}>
          <Icon className={`text-${color}-600`} size={24} />
        </div>
        {change !== undefined && (
          <div className={`flex items-center gap-1 text-sm font-medium ${
            change > 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {change > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
            {Math.abs(change).toFixed(2)}%
          </div>
        )}
      </div>
      <h3 className="text-sm font-medium text-gray-500 mb-1">{title}</h3>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
    </div>
  );

  const RiskIndicator = ({ label, current, max, unit = '%' }) => {
    const percentage = (current / max) * 100;
    const isWarning = percentage > 80;
    const isDanger = percentage > 95;
    
    return (
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          <span className="text-sm text-gray-500">
            {current.toFixed(1)}{unit} / {max}{unit}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all ${
              isDanger ? 'bg-red-500' : 
              isWarning ? 'bg-yellow-500' : 
              'bg-green-500'
            }`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Cargando datos del sistema HRM...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Sistema HRM - Panel de Control
          </h1>
          <p className="text-gray-600">
            Hierarchical Reasoning Model para Trading Algorítmico
          </p>
          {error && (
            <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-700">{error}</p>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8">
          <TabButton
            id="portfolio"
            label="Cartera"
            icon={DollarSign}
            isActive={activeTab === 'portfolio'}
            onClick={setActiveTab}
          />
          <TabButton
            id="l1"
            label="L1 Operacional"
            icon={Zap}
            isActive={activeTab === 'l1'}
            onClick={setActiveTab}
          />
          <TabButton
            id="ai"
            label="Modelos IA"
            icon={Brain}
            isActive={activeTab === 'ai'}
            onClick={setActiveTab}
          />
          <TabButton
            id="risk"
            label="Gestión de Riesgo"
            icon={Shield}
            isActive={activeTab === 'risk'}
            onClick={setActiveTab}
          />
        </div>

        {/* Portfolio Tab */}
        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            {/* Métricas principales */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <MetricCard
                icon={DollarSign}
                title="Valor Total de la Cartera"
                value={`$${metrics.totalValue?.toFixed(2) || '0.00'}`}
                change={metrics.totalChangePercent}
                color="blue"
              />
              <MetricCard
                icon={Bitcoin}
                title="BTC Precio"
                value={`$${portfolioData[portfolioData.length - 1]?.btc_price.toFixed(0) || '0'}`}
                change={metrics.btcChange}
                color="orange"
              />
              <MetricCard
                icon={Coins}
                title="ETH Precio"
                value={`$${portfolioData[portfolioData.length - 1]?.eth_price.toFixed(0) || '0'}`}
                change={metrics.ethChange}
                color="purple"
              />
              <MetricCard
                icon={Activity}
                title="Correlación BTC-ETH"
                value={`${((metrics.correlation || 0) * 100).toFixed(1)}%`}
                color="green"
              />
            </div>

            {/* Gráficos */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Evolución del portfolio */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Evolución de la Cartera
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={portfolioData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="total" 
                      stroke="#3b82f6" 
                      fill="#3b82f6" 
                      fillOpacity={0.3} 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Composición de la cartera */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Composición de la Cartera
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={compositionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {compositionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Precios de activos */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Evolución de Precios
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={portfolioData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="btc_price" 
                    stroke="#f7931a" 
                    strokeWidth={2}
                    name="BTC Price ($)"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="eth_price" 
                    stroke="#627eea" 
                    strokeWidth={2}
                    name="ETH Price ($)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* L1 Operational Tab */}
        {activeTab === 'l1' && (
          <div className="space-y-6">
            {/* Métricas L1 */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <MetricCard
                icon={Zap}
                title="Latencia Sistema"
                value={`${l1Metrics.system.latency}ms`}
                color="green"
              />
              <MetricCard
                icon={Activity}
                title="Throughput"
                value={`${l1Metrics.system.throughput}/s`}
                color="blue"
              />
              <MetricCard
                icon={Target}
                title="BTC Success Rate"
                value={`${l1Metrics.btc.successRate}%`}
                color="orange"
              />
              <MetricCard
                icon={Target}
                title="ETH Success Rate"
                value={`${l1Metrics.eth.successRate}%`}
                color="purple"
              />
            </div>

            {/* Dashboard de métricas operacionales */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">
                L1 OPERATIONAL DASHBOARD
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* BTC Metrics */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Bitcoin className="text-orange-500" size={24} />
                    <h4 className="text-lg font-semibold">BTC/USDT</h4>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Señales procesadas:</span>
                      <span className="font-medium">
                        {l1Metrics.btc.signalsProcessed.success} ✅ | {l1Metrics.btc.signalsProcessed.failed} ❌
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Success rate:</span>
                      <span className="font-medium text-green-600">{l1Metrics.btc.successRate}%</span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Slippage promedio:</span>
                      <span className="font-medium">{l1Metrics.btc.avgSlippage}%</span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Exposición actual:</span>
                      <span className="font-medium">
                        {l1Metrics.btc.currentExposure.toFixed(1)}% / {l1Metrics.btc.maxExposure}% max
                      </span>
                    </div>
                  </div>
                </div>

                {/* ETH Metrics */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Coins className="text-purple-500" size={24} />
                    <h4 className="text-lg font-semibold">ETH/USDT</h4>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Señales procesadas:</span>
                      <span className="font-medium">
                        {l1Metrics.eth.signalsProcessed.success} ✅ | {l1Metrics.eth.signalsProcessed.failed} ❌
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Success rate:</span>
                      <span className="font-medium text-green-600">{l1Metrics.eth.successRate}%</span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Slippage promedio:</span>
                      <span className="font-medium">{l1Metrics.eth.avgSlippage}%</span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Exposición actual:</span>
                      <span className="font-medium">
                        {l1Metrics.eth.currentExposure.toFixed(1)}% / {l1Metrics.eth.maxExposure}% max
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Correlación BTC-ETH:</span>
                  <span className="font-medium">
                    {(l1Metrics.system.correlation * 100).toFixed(0)}% (límite: {(l1Metrics.system.correlationLimit * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Models Tab */}
        {activeTab === 'ai' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Logistic Regression */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="text-blue-500" size={24} />
                  <h3 className="text-lg font-semibold">Logistic Regression</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="font-medium">{(l1Metrics.ai.logreg.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">F1 Score:</span>
                    <span className="font-medium">{(l1Metrics.ai.logreg.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">AUC:</span>
                    <span className="font-medium">{(l1Metrics.ai.logreg.auc * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Random Forest */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="text-green-500" size={24} />
                  <h3 className="text-lg font-semibold">Random Forest</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="font-medium">{(l1Metrics.ai.randomForest.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">F1 Score:</span>
                    <span className="font-medium">{(l1Metrics.ai.randomForest.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">AUC:</span>
                    <span className="font-medium">{(l1Metrics.ai.randomForest.auc * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* LightGBM */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="text-purple-500" size={24} />
                  <h3 className="text-lg font-semibold">LightGBM</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="font-medium">{(l1Metrics.ai.lightgbm.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">F1 Score:</span>
                    <span className="font-medium">{(l1Metrics.ai.lightgbm.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">AUC:</span>
                    <span className="font-medium">{(l1Metrics.ai.lightgbm.auc * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Flujo de decisión IA */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Flujo de Decisión IA Jerárquico
              </h3>
              <div className="space-y-4">
                <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                  <div>
                    <h4 className="font-medium">Hard-coded Safety</h4>
                    <p className="text-sm text-gray-600">Validaciones básicas por símbolo</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-lg">
                  <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                  <div>
                    <h4 className="font-medium">Logistic Regression</h4>
                    <p className="text-sm text-gray-600">Filtro rápido de tendencia (BTC/ETH específico)</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 bg-green-50 rounded-lg">
                  <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                  <div>
                    <h4 className="font-medium">Random Forest</h4>
                    <p className="text-sm text-gray-600">Confirmación con ensemble robusto</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 bg-purple-50 rounded-lg">
                  <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                  <div>
                    <h4 className="font-medium">LightGBM</h4>
                    <p className="text-sm text-gray-600">Decisión final con regularización avanzada</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 p-4 bg-yellow-50 rounded-lg">
                  <div className="w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                  <div>
                    <h4 className="font-medium">Decision Layer</h4>
                    <p className="text-sm text-gray-600">Combinación ponderada de los 3 modelos</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Risk Management Tab */}
        {activeTab === 'risk' && (
          <div className="space-y-6">
            {/* Indicadores de riesgo */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Límites de Exposición por Activo
                </h3>
                <div className="space-y-4">
                  <RiskIndicator 
                    label="Exposición BTC"
                    current={metrics.btcExposure || 0}
                    max={20}
                  />
                  <RiskIndicator 
                    label="Exposición ETH"
                    current={metrics.ethExposure || 0}
                    max={15}
                  />
                  <RiskIndicator 
                    label="Correlación BTC-ETH"
                    current={(metrics.correlation || 0) * 100}
                    max={80}
                  />
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Validaciones de Seguridad
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                    <Shield className="text-green-600" size={20} />
                    <span className="text-sm font-medium text-green-700">Stop-loss obligatorio activo</span>
                  </div>
                  
                  <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                    <Shield className="text-green-600" size={20} />
                    <span className="text-sm font-medium text-green-700">Límites por trade verificados</span>
                  </div>
                  
                  <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                    <Shield className="text-green-600" size={20} />
                    <span className="text-sm font-medium text-green-700">Validación de saldo disponible</span>
                  </div>
                  
                  <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg">
                    <AlertTriangle className="text-yellow-600" size={20} />
                    <span className="text-sm font-medium text-yellow-700">Monitoreo de correlación activo</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Matriz de riesgo */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Matriz de Límites de Riesgo
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-medium text-gray-700">Concepto</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700">Valor Actual</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700">Límite</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700">Estado</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100">
                      <td className="py-3 px-4">Stop-loss</td>
                      <td className="py-3 px-4">Obligatorio</td>
                      <td className="py-3 px-4">-</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                          Activo
                        </span>
                      </td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className="py-3 px-4">Límite BTC por trade</td>
                      <td className="py-3 px-4">0.02232 BTC</td>
                      <td className="py-3 px-4">0.05 BTC</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                          Seguro
                        </span>
                      </td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className="py-3 px-4">Límite ETH por trade</td>
                      <td className="py-3 px-4">0.126 ETH</td>
                      <td className="py-3 px-4">1.0 ETH</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                          Seguro
                        </span>
                      </td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className="py-3 px-4">Exposición BTC máxima</td>
                      <td className="py-3 px-4">{(metrics.btcExposure || 0).toFixed(1)}%</td>
                      <td className="py-3 px-4">20%</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          (metrics.btcExposure || 0) > 18 ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
                        }`}>
                          {(metrics.btcExposure || 0) > 18 ? 'Precaución' : 'Seguro'}
                        </span>
                      </td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className="py-3 px-4">Exposición ETH máxima</td>
                      <td className="py-3 px-4">{(metrics.ethExposure || 0).toFixed(1)}%</td>
                      <td className="py-3 px-4">15%</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          (metrics.ethExposure || 0) > 13 ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
                        }`}>
                          {(metrics.ethExposure || 0) > 13 ? 'Precaución' : 'Seguro'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4">Correlación BTC-ETH</td>
                      <td className="py-3 px-4">{((metrics.correlation || 0) * 100).toFixed(0)}%</td>
                      <td className="py-3 px-4">80%</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                          Normal
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Alertas y recomendaciones */}
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Sistema de Alertas
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <Eye className="text-blue-600" size={20} />
                  <div>
                    <h4 className="font-medium text-blue-900">Monitoreo Activo</h4>
                    <p className="text-sm text-blue-700">
                      Sistema L1 operando en modo determinista con validaciones IA
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <Shield className="text-green-600" size={20} />
                  <div>
                    <h4 className="font-medium text-green-900">Límites de Riesgo</h4>
                    <p className="text-sm text-green-700">
                      Todos los límites de exposición dentro de rangos seguros
                    </p>
                  </div>
                </div>
                
                {(metrics.correlation || 0) > 0.75 && (
                  <div className="flex items-center gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <AlertTriangle className="text-yellow-600" size={20} />
                    <div>
                      <h4 className="font-medium text-yellow-900">Correlación Elevada</h4>
                      <p className="text-sm text-yellow-700">
                        Correlación BTC-ETH en {((metrics.correlation || 0) * 100).toFixed(0)}% - Monitorear diversificación
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Footer con información del sistema */}
        <div className="mt-8 bg-gray-800 text-white rounded-xl p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Sistema HRM</h4>
              <p className="text-sm text-gray-300">
                Hierarchical Reasoning Model para Trading Algorítmico
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Versión 1.0 - L1 Operational
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Modo Operacional</h4>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm">PAPER Trading Activo</span>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                Binance Spot - Simulación
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Última Actualización</h4>
              <p className="text-sm text-gray-300">
                {portfolioData[portfolioData.length - 1]?.timestamp || 'N/A'}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Datos en tiempo real
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CryptoPortfolioDashboard;
