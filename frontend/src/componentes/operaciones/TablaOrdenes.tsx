import React from 'react';
import {
  claseLado,
  etiquetaEstadoOrden,
  etiquetaLado,
  formatearCantidadOrden,
  formatearFechaOrden,
  formatearPrecioOrden,
  type Orden,
} from '../../dominio';

const COLUMNAS = ['Fecha', 'Símbolo', 'Lado', 'Cantidad', 'Precio', 'Comisión', 'Estado', 'Estrategia'];

const CabeceraOrdenes: React.FC = () => (
  <thead className="bg-gray-50">
    <tr>
      {COLUMNAS.map((titulo) => (
        <th key={titulo} scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
          {titulo}
        </th>
      ))}
    </tr>
  </thead>
);

const FilaOrden: React.FC<{ orden: Orden }> = ({ orden }) => (
  <tr className="hover:bg-gray-50">
    <td className="px-4 py-3 text-sm text-gray-600">{formatearFechaOrden(orden.timestamp)}</td>
    <td className="px-4 py-3 text-sm font-medium text-gray-900">{orden.simbolo}</td>
    <td className="px-4 py-3">
      <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-medium ${claseLado(orden.lado)}`}>
        {etiquetaLado(orden.lado)}
      </span>
    </td>
    <td className="px-4 py-3 text-sm text-gray-600">{formatearCantidadOrden(orden.cantidad)}</td>
    <td className="px-4 py-3 text-sm text-gray-600">{formatearPrecioOrden(orden.precio_llenado)}</td>
    <td className="px-4 py-3 text-sm text-gray-600">{formatearPrecioOrden(orden.comision)}</td>
    <td className="px-4 py-3 text-sm text-gray-600">{etiquetaEstadoOrden(orden.status)}</td>
    <td className="px-4 py-3 text-sm text-gray-600">{orden.estrategia ?? '—'}</td>
  </tr>
);

export const TablaOrdenes: React.FC<{ ordenes: Orden[] }> = ({ ordenes }) => (
  <div className="overflow-x-auto">
    <table className="min-w-full divide-y divide-gray-200">
      <CabeceraOrdenes />
      <tbody className="bg-white divide-y divide-gray-200">
        {ordenes.map((orden) => (
          <FilaOrden key={orden.order_id} orden={orden} />
        ))}
      </tbody>
    </table>
  </div>
);
