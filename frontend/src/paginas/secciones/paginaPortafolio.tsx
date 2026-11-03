import React from 'react';
import { useModoStore } from '../../estado/storeModo';

export const PaginaPortafolio: React.FC = () => {
  const { modo } = useModoStore();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Portafolio</h1>
      <p className="mt-2 text-gray-600">Estado del modo: <span className="font-semibold">{modo.toUpperCase()}</span></p>
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {['Monedas', 'Asignaciones', 'Saldo total'].map((item) => (
          <div key={item} className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold">{item}</h2>
            <p className="mt-2 text-sm text-gray-600">Información del portafolio en modo {modo}.</p>
          </div>
        ))}
      </div>
    </div>
  );
};
