import React from 'react';
import { usarEtiquetaModo } from '../../hooks/useEtiquetaModo';

export const PaginaReportes: React.FC = () => {
  const etiquetaModo = usarEtiquetaModo();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Reportes</h1>
      <p className="mt-2 text-gray-600">Estado del modo: <span className="font-semibold">{etiquetaModo}</span></p>
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {['Resumen de operaciones', 'Estado de riesgo', 'Arbol de performace'].map((titulo) => (
          <div key={titulo} className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold">{titulo}</h2>
            <p className="mt-2 text-sm text-gray-600">Sin datos: este reporte aún no está conectado al backend.</p>
          </div>
        ))}
      </div>
    </div>
  );
};
