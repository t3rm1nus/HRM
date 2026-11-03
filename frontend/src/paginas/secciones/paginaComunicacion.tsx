import React from 'react';
import { useModoStore } from '../../estado/storeModo';

export const PaginaComunicacion: React.FC = () => {
  const { modo } = useModoStore();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Comunicación</h1>
      <p className="mt-2 text-gray-600">Estado del modo: <span className="font-semibold">{modo.toUpperCase()}</span></p>
      <div className="mt-6 bg-white border border-gray-200 rounded-lg p-6">
        <h2 className="text-lg font-semibold">Comunicación en tiempo real</h2>
        <p className="mt-2 text-sm text-gray-600">Sección de comunicación en modo {modo}.</p>
      </div>
    </div>
  );
};
