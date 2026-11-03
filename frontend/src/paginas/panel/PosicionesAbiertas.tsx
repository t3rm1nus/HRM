import React from 'react';
import { TarjetaKpi } from '../ui/tarjetaKpi';

export interface PosicionAbierta {
  id: string;
  simbolo: string;
  tipo: 'comprador' | 'vendedor';
  cantidad: number;
  precioEntrada: number;
  precioActual: number;
  pnl: number;
  pnlPorcentaje: number;
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
}

export const PosicionesAbiertas: React.FC = () => {
  const posiciones: PosicionAbierta[] = [
    {
      id: '1',
      simbolo: 'BTC/USD',
      tipo: 'comprador',
      cantidad: 0.045,
      precioEntrada: 65230,
      precioActual: 67890,
      pnl: 1320,
      pnlPorcentaje: 2.02,
      regimen: 'LIVE',
    },
    {
      id: '2',
      simbolo: 'ETH/USD',
      tipo: 'vendedor',
      cantidad: 3.2,
      precioEntrada: 3800,
      precioActual: 3745,
      pnl: -175,
      pnlPorcentaje: -4.61,
      regimen: 'LIVE',
    },
  ];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm">
      <h2 className="text-lg font-medium text-gray-600 mb-4">Posiciones Abiertas</h2>
      {posiciones.map((pos) => (
        <div key={pos.id} className="p-4 border rounded-lg mb-3" style={{ borderColor: pos.pnl >= 0 ? 'border-green-500' : 'border-red-500' }}>
          <div className="flex justify-between items-start">
            <span className="font-medium text-gray-900">{pos.simbolo}</span>
            <span className="text-sm text-gray-500">{pos.regimen}</span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-sm font-medium">
              {pos.tipo === 'comprador' ? 'Largo' : 'Corto'}
            </span>
            <span className="text-xs text-gray-500">×{pos.cantidad}</span>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-sm font-medium">
              ${pos.precioEntrada.toLocaleString('es-ES')}
            </span>
            <span className="text-sm font-medium ${pos.pnl >= 0 ? 'text-green-600' : 'text-red-600'}">
              ${pos.pnlPorcentaje > 0 ? '+' : ''}${pos.pnlPorcentaje.toFixed(2)}%
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-xs text-gray-500">PnL: ${pos.pnl.toLocaleString('es-ES')}</span>
            {pos.pnl >= 0 && (
              <span className="text-xs text-green-600 font-medium">✓</span>
            )}
            {pos.pnl < 0 && (
              <span className="text-xs text-red-600 font-medium">✗</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default PosicionesAbiertas;