import React from 'react';
import { EstadoVacio } from '../../componentes/ui/estadoVacio';

export const MENSAJE_OPERACIONES_SIN_DATOS =
  'Sin datos: el listado de operaciones aún no está conectado al backend.';

export const DETALLE_OPERACIONES_SIN_DATOS =
  'Cuando exista un endpoint de operaciones, aquí se mostrarán las ejecutadas, rechazadas y bloqueadas.';

export const PaginaOperaciones: React.FC = () => (
  <div className="max-w-7xl mx-auto px-4 py-8">
    <h1 className="text-2xl font-bold text-gray-900">Operaciones</h1>
    <div className="mt-6 bg-white border border-gray-200 rounded-lg">
      <EstadoVacio
        mensaje={MENSAJE_OPERACIONES_SIN_DATOS}
        detalle={DETALLE_OPERACIONES_SIN_DATOS}
      />
    </div>
  </div>
);
