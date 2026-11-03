import React from 'react';

export interface Senal {
  id: string;
  tipo: 'compra' | 'venta' | 'take-profit' | 'stop-loss';
  simbolo: string;
  precio: number;
  timestamp: string;
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
}

export interface Orden {
  id: string;
  simbolo: string;
  tipo: 'compra' | 'venta';
  cantidad: number;
  precio: number;
  estatus: 'pendiente' | 'ejecutada' | 'cancelada';
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
}

export const SenalesOrdenes: React.FC = () => {
  const senales: Senal[] = [
    {
      id: '1',
      tipo: 'compra',
      simbolo: 'BTC/USD',
      precio: 67890,
      timestamp: 'hace 5m',
      regimen: 'LIVE',
    },
    {
      id: '2',
      tipo: 'take-profit',
      simbolo: 'ETH/USD',
      precio: 3890,
      timestamp: 'hace 15m',
      regimen: 'LIVE',
    },
  ];

  const ordenes: Orden[] = [
    {
      id: '3',
      simbolo: 'BTC/USD',
      tipo: 'compra',
      cantidad: 0.01,
      precio: 67500,
      estatus: 'pendiente',
      regimen: 'LIVE',
    },
    {
      id: '4',
      simbolo: 'ETH/USD',
      tipo: 'venta',
      cantidad: 2,
      precio: 3800,
      estatus: 'ejecutada',
      regimen: 'LIVE',
    },
  ];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm">
      <h2 className="text-lg font-medium text-gray-600 mb-4">Últimas Señales y Órdenes</h2>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <h3 className="text-sm font-medium text-gray-600 mb-2">Señales</h3>
        {senales.map((senal) => (
          <div key={senal.id} className="p-3 border rounded-lg text-sm">
            <span className={`font-medium ${senal.tipo === 'compra' ? 'text-green-600' : 'text-red-600'}`}>
              {senal.tipo}
            </span>
            <span className="text-gray-500 ml-2">↑${senal.precio}</span>
            <span className="text-xs text-gray-400 ml-2">{senal.regimen}</span>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <h3 className="text-sm font-medium text-gray-600 mb-2">Órdenes</h3>
        {ordenes.map((orden) => (
          <div key={orden.id} className="p-3 border rounded-lg text-sm">
            <span className={`font-medium ${orden.estatus === 'pendiente' ? 'text-yellow-600' : orden.estatus === 'ejecutada' ? 'text-green-600' : 'text-red-600'}`}>
              {orden.estatus}
            </span>
            <span className="text-gray-500 ml-2">×{orden.cantidad}</span>
            <span className="text-xs text-gray-400 ml-2">{orden.regimen}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SenalesOrdenes;