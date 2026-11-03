import React from 'react';

const LIMITES = [
  { nombre: 'Drawdown', uso: 0.9, limite: 1.0 },
  { nombre: 'Exposición', uso: 0.6, limite: 1.0 },
  { nombre: 'Tamaño', uso: 0.4, limite: 1.0 },
  { nombre: 'Reserva USDT', uso: 0.3, limite: 1.0 },
];

export const PaginaRiesgo: React.FC = () => {
  const [alertas, setAlertas] = React.useState([
    { id: '1', severidad: 'alta', mensaje: 'Drawdown supera 85%' },
    { id: '2', severidad: 'media', mensaje: 'Exposición BTC alta' },
  ]);
  const [filtro, setFiltro] = React.useState('');
  const reconocer = (id: string) => setAlertas((a) => a.filter((x) => x.id !== id));
  const filtradas = alertas.filter((a) => (filtro ? a.severidad === filtro : true));
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Riesgo</h1>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {LIMITES.map((l) => {
          const pct = l.uso / l.limite;
          const rojo = pct >= 0.85;
          return (
            <div key={l.nombre} className="bg-white border rounded-xl p-4">
              <p className="text-sm font-semibold">{l.nombre}</p>
              <div className="mt-2 h-3 bg-gray-200 rounded">
                <div className={`h-3 rounded ${rojo ? 'bg-red-500' : 'bg-green-500'}`} style={{ width: `${pct * 100}%` }} />
              </div>
              <p className={`text-xs mt-1 ${rojo ? 'text-red-600' : 'text-gray-600'}`}>{(pct * 100).toFixed(0)}%</p>
            </div>
          );
        })}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Stop-loss activos</h2>
        <ul className="text-sm">
          <li>BTCUSDT - SL 2%</li>
          <li>ETHUSDT - SL 2.5%</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Circuit breakers</h2>
        <ul className="text-sm">
          <li>Drawdown diario: umbral 8%</li>
          <li>Volatilidad: umbral 5%</li>
        </ul>
      </div>
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">VaR / CVaR</h2>
          <p className="text-sm">VaR 95%: -2.3%</p>
          <p className="text-sm">CVaR 95%: -3.1%</p>
        </div>
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Matriz de correlación</h2>
          <p className="text-sm text-gray-600">BTC-ETH: 0.72</p>
        </div>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Escenarios de estrés</h2>
        <ul className="text-sm">
          <li>Crash -20%: -4.5%</li>
          <li>Rally +15%: +3.2%</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Histórico rechazos RiskGuard</h2>
        <ul className="text-sm">
          <li>2026-02-12 - orden excede tamaño</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold">Feed de alertas</h2>
          <select value={filtro} onChange={(e) => setFiltro(e.target.value)} className="px-2 py-1 border rounded text-sm">
            <option value="">Todas</option>
            <option value="alta">Alta</option>
            <option value="media">Media</option>
          </select>
        </div>
        <ul className="text-sm space-y-2">
          {filtradas.map((a) => (
            <li key={a.id} className="flex items-center justify-between">
              <span className={a.severidad === 'alta' ? 'text-red-600' : 'text-yellow-600'}>
                [{a.severidad}] {a.mensaje}
              </span>
              <button type="button" onClick={() => reconocer(a.id)} className="text-xs text-blue-600 hover:underline">
                Reconocer
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
