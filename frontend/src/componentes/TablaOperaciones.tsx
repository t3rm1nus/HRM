import React from 'react';
import { TipoOperacion, ColorTipo, ColorEstado, TextoEstado } from '../constantes';

export interface Operacion {
  id: string;
  usuario: string;
  tipo: TipoOperacion;
  monto: number;
  fecha: string;
  estado: string;
  descripcion: string;
}

export const FilaOperacion: React.FC<{ operacion: Operacion }> = ({ operacion }) => {
  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      <td className="px-4 py-3 text-sm text-gray-600">{operacion.id}</td>
      <td className="px-4 py-3 text-sm text-gray-600">{operacion.usuario}</td>
      <td className="px-4 py-3">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${ColorTipo(operacion.tipo)}`}>
          {operacion.tipo}
        </span>
      </td>
      <td className="px-4 py-3 text-sm text-gray-600">
        {new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(operacion.monto)}
      </td>
      <td className="px-4 py-3 text-sm text-gray-600">{new Date(operacion.fecha).toLocaleDateString('es-ES')}</td>
      <td className="px-4 py-3">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${ColorEstado(operacion.estado)}`}>
          {TextoEstado(operacion.estado)}
        </span>
      </td>
    </tr>
  );
};

export const CabeceraTabla: React.FC = () => {
  return (
    <thead className="bg-gray-50">
      <tr>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Usuario</th>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Tipo</th>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Monto</th>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Fecha</th>
        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Estado</th>
      </tr>
    </thead>
  );
};

export const TablaOperaciones: React.FC<{ operaciones: Operacion[] }> = ({ operaciones }) => {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <CabeceraTabla />
        <tbody className="bg-white divide-y divide-gray-200">
          {operaciones.map((operacion) => (
            <FilaOperacion key={operacion.id} operacion={operacion} />
          ))}
        </tbody>
      </table>
    </div>
  );
};