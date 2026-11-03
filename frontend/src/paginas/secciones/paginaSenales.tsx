import React from 'react';

interface Modelo {
  nombre: string;
  fuente: string;
  fuerza: number | null;
  confianza: number | null;
  causa?: string;
}

const MODELOS: Modelo[] = [
  { nombre: 'PPO-1', fuente: 'ppo', fuerza: 0.8, confianza: 0.7 },
  { nombre: 'PPO-2', fuente: 'ppo', fuerza: 0.6, confianza: 0.65 },
  { nombre: 'PPO-3', fuente: 'ppo', fuerza: null, confianza: null, causa: 'sin datos recientes' },
  { nombre: 'PPO-4', fuente: 'ppo', fuerza: 0.5, confianza: 0.6 },
  { nombre: 'PPO-5', fuente: 'ppo', fuerza: 0.7, confianza: 0.72 },
  { nombre: 'PPO-6', fuente: 'ppo', fuerza: 0.4, confianza: 0.55 },
  { nombre: 'PPO-7', fuente: 'ppo', fuerza: 0.9, confianza: 0.8 },
  { nombre: 'Tecnico', fuente: 'tecnico', fuerza: 0.65, confianza: 0.7 },
  { nombre: 'Patrones', fuente: 'patrones', fuerza: 0.55, confianza: 0.6 },
];

export const PaginaSenales: React.FC = () => {
  const generadas = 120;
  const validadas = 80;
  const ejecutadas = 45;
  const ensamble = MODELOS.filter((m) => m.fuerza !== null).reduce((a, m) => a + (m.fuerza ?? 0), 0) / MODELOS.filter((m) => m.fuerza !== null).length;
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Señales</h1>
      <p className="mt-2 text-sm text-gray-600">Modo PATH activo: L3 estrategico</p>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
        <div className="bg-white border rounded-xl p-4 text-center">
          <p className="text-xs text-gray-500">Generadas</p>
          <p className="text-xl font-bold">{generadas}</p>
        </div>
        <div className="bg-white border rounded-xl p-4 text-center">
          <p className="text-xs text-gray-500">Validadas</p>
          <p className="text-xl font-bold">{validadas}</p>
        </div>
        <div className="bg-white border rounded-xl p-4 text-center">
          <p className="text-xs text-gray-500">Ejecutadas</p>
          <p className="text-xl font-bold">{ejecutadas}</p>
        </div>
      </div>
      <p className="mt-2 text-sm text-gray-600">Votacion ensemble: {ensamble.toFixed(2)}</p>
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {MODELOS.map((m) => (
          <div
            key={m.nombre}
            className={`border rounded-xl p-4 ${m.fuerza === null ? 'bg-gray-100 text-gray-400' : 'bg-white'}`}
          >
            <p className="font-semibold">{m.nombre}</p>
            <p className="text-xs">{m.fuente}</p>
            {m.fuerza === null ? (
              <p className="text-xs mt-2">Ausente: {m.causa}</p>
            ) : (
              <>
                <p className="text-sm">Fuerza: {m.fuerza.toFixed(2)}</p>
                <p className="text-sm">Confianza: {m.confianza?.toFixed(2)}</p>
              </>
            )}
          </div>
        ))}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Señales en espera</h2>
        <ul className="text-sm">
          <li>BTCUSDT - momentum - pendiente validacion</li>
          <li>ETHUSDT - mean_reversion - pendiente ejecucion</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Detector de similitud</h2>
        <p className="text-sm text-gray-600">3 señales similares detectadas en ultimas 24h</p>
      </div>
    </div>
  );
};
