import React from 'react';
import { useModoStore } from '../../estado/storeModo';
import { ModoAplicacion } from '../../dominio/modo';

const MECANISMOS = [
  'Validación de señal',
  'Control de confianza',
  'Filtro de régimen',
  'Gestión de riesgo',
  'Stop-loss dinámico',
  'Take-profit escalonado',
  'Cooldown adaptativo',
  'Rebalanceo automático',
  'Auditoría de decisiones',
];

export const PaginaAutoaprendizaje: React.FC = () => {
  const { modo } = useModoStore();
  const buffer = 320;
  const minimo = 500;
  const porcentaje = Math.min(100, (buffer / minimo) * 100);
  const [historial] = React.useState([
    { fecha: '2026-02-10', resultado: 'exitoso' },
    { fecha: '2026-02-05', resultado: 'fallido' },
  ]);
  const reentrenar = () => {
    if (modo !== ModoAplicacion.PAPERA) return;
    if (window.confirm('¿Reentrenar manualmente?')) {
      // noop mock
    }
  };
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Autoaprendizaje</h1>
      <p className="mt-2 text-sm text-gray-600">Las «9 IAs» son mecanismos de protección, no nueve modelos.</p>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Mecanismos (9)</h2>
        <ul className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
          {MECANISMOS.map((m) => (
            <li key={m} className="bg-gray-50 rounded p-2">{m}: activo</li>
          ))}
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Buffer de trades</h2>
        <div className="h-3 bg-gray-200 rounded">
          <div className="h-3 bg-blue-500 rounded" style={{ width: `${porcentaje}%` }} />
        </div>
        <p className="text-xs mt-1">{buffer}/{minimo}</p>
      </div>
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Tamaño del ensemble</h2>
          <p className="text-sm">7 modelos</p>
        </div>
        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Detección de drift</h2>
          <p className="text-sm">Sin drift detectado</p>
        </div>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Historial de reentrenamientos</h2>
        <ul className="text-sm">
          {historial.map((h, i) => (
            <li key={i}>{h.fecha} - {h.resultado}</li>
          ))}
        </ul>
      </div>
      <div className="mt-6">
        <button
          type="button"
          disabled={modo !== ModoAplicacion.PAPERA}
          onClick={reentrenar}
          className={`px-4 py-2 rounded-lg text-white ${modo !== ModoAplicacion.PAPERA ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
        >
          Reentrenar manualmente
        </button>
      </div>
    </div>
  );
};
