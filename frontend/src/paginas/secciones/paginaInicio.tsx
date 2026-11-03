import React from 'react';
import { useModoStore } from '../../estado/storeModo';
import { PAGINAS } from '../../constantes/paginas';

export const PaginaInicial: React.FC = () => {
  const { modo } = useModoStore();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white rounded-xl shadow p-6">
        <h1 className="text-2xl font-bold text-gray-900">Panel de Control</h1>
        <p className="mt-2 text-gray-600">
          Bienvenido al sistema HRM. Modo actual: <span className="font-semibold">{modo.toUpperCase()}</span>
        </p>
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PAGINAS.slice(0, 4).map((pg) => (
            <div key={pg.ruta} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-blue-600">{pg.icono}</div>
              <p className="mt-2 font-medium text-gray-900">{pg.nombre}</p>
              <a href={pg.ruta} className="text-sm text-blue-600 hover:underline">Ver {pg.nombre}</a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
