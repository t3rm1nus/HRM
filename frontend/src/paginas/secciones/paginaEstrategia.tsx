import React from 'react';
import { useSearchParams } from 'react-router-dom';

export const PaginaEstrategia: React.FC = () => {
  const [params] = useSearchParams();
  const estado = params.get('estado') ?? 'normal';
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Estrategia</h1>
      <div className="mt-4 bg-white border rounded-xl p-4">
        <p className="font-semibold">Régimen: {estado === 'ciego' ? 'unknown_no_sync' : 'bullish'}</p>
        <p className="text-sm text-gray-600">Subtipo: {estado === 'sintetico' ? 'sintético' : 'tendencial'}</p>
        <p className="text-sm text-gray-600">Confianza: {estado === 'ciego' ? 'n/a' : '0.82'}</p>
        {estado === 'sintetico' && (
          <span className="inline-block mt-2 px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded">SINTÉTICO</span>
        )}
        {estado === 'ciego' && (
          <span className="inline-block mt-2 px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded">Modo ciego (unknown_no_sync)</span>
        )}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Línea temporal de regímenes</h2>
        <div className="h-24 bg-gray-100 rounded flex items-center justify-center text-sm text-gray-500">[grafico precio/regimen]</div>
      </div>
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Sentimiento</h2>
          <p className="text-sm">Medidor: {estado === 'sintetico' ? '0.6 (sintético)' : '0.7'}</p>
        </div>
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Volatilidad (GARCH/LSTM)</h2>
          <p className="text-sm">Previsión: 0.024</p>
        </div>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Apetito de riesgo</h2>
        <p className="text-sm">Moderado</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Pesos Black-Litterman</h2>
        <ul className="text-sm">
          <li>BTC: 40%</li>
          <li>ETH: 30%</li>
          <li>USDT: 30%</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Historial decisiones L3</h2>
        <table className="min-w-full text-sm">
          <thead>
            <tr><th className="text-left">Fecha</th><th className="text-left">Decisión</th></tr>
          </thead>
          <tbody>
            <tr><td>2026-02-12</td><td>Rebalancear hacia BTC</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
