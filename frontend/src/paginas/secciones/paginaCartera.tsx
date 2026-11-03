import React from 'react';
import { calcularDesviacion } from '../../dominio/cartera';

const ASIGNACION = [
  { activo: 'BTC', objetivo: 0.4, real: 0.35 },
  { activo: 'ETH', objetivo: 0.3, real: 0.32 },
  { activo: 'USDT', objetivo: 0.3, real: 0.33 },
];

const LOTES = [
  { id: 'L1', activo: 'BTC', cantidad: 0.0012, precio: 65000, fecha: '2026-02-10' },
  { id: 'L2', activo: 'ETH', cantidad: 0.05, precio: 1900, fecha: '2026-02-11' },
];

const REBALANCEOS = [
  { fecha: '2026-02-12', motivo: 'Desviación > 5%', accion: 'Comprar BTC' },
];

export const PaginaCartera: React.FC = () => {
  const esRespaldo = true;
  const desviacion = calcularDesviacion(ASIGNACION);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Cartera</h1>
      {esRespaldo && (
        <div className="mt-4 p-3 bg-red-100 text-red-800 rounded-lg">
          Aviso: balances de respaldo
        </div>
      )}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Donut de asignación</h2>
          <div className="flex gap-2">
            {ASIGNACION.map((a) => (
              <div key={a.activo} className="flex-1 text-center">
                <div className="h-2 bg-blue-500 rounded" style={{ width: `${a.real * 100}%` }} />
                <p className="text-xs mt-1">{a.activo} {(a.real * 100).toFixed(0)}%</p>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Exposición vs límites</h2>
          {ASIGNACION.map((a) => (
            <div key={a.activo} className="mb-2">
              <div className="flex justify-between text-xs">
                <span>{a.activo}</span>
                <span>{(a.real * 100).toFixed(0)}% / 50%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded">
                <div className="h-2 bg-green-500 rounded" style={{ width: `${Math.min(a.real * 100, 50)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Asignación objetivo vs real (desviación: {desviacion.toFixed(3)})</h2>
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="text-left">Activo</th>
              <th className="text-left">Objetivo</th>
              <th className="text-left">Real</th>
              <th className="text-left">Desv.</th>
            </tr>
          </thead>
          <tbody>
            {ASIGNACION.map((a) => (
              <tr key={a.activo}>
                <td>{a.activo}</td>
                <td>{(a.objetivo * 100).toFixed(0)}%</td>
                <td>{(a.real * 100).toFixed(0)}%</td>
                <td>{Math.abs(a.objetivo - a.real).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Lotes FIFO abiertos</h2>
        <ul className="text-sm">
          {LOTES.map((l) => (
            <li key={l.id}>{l.activo} {l.cantidad} @ {l.precio} ({l.fecha})</li>
          ))}
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Historial de rebalanceos</h2>
        <ul className="text-sm">
          {REBALANCEOS.map((r, i) => (
            <li key={i}>{r.fecha} - {r.motivo} - {r.accion}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};
