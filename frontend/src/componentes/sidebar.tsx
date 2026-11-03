import React from 'react';
import { NavLink } from 'react-router-dom';
import { useModoStore } from '../estado/storeModo';
import { PAGINAS } from '../constantes/paginas';

export const Sidebar: React.FC = () => {
  const { modo } = useModoStore();

  return (
    <aside className="flex h-full flex-col border-r border-gray-200 bg-white">
      <div className="flex items-center gap-3 px-4 py-5">
        <span className="text-2xl font-bold">HRM</span>
        <span className="text-sm text-gray-500">Trading</span>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
        {PAGINAS.map((pg) => (
          <NavLink
            key={pg.ruta}
            to={pg.ruta}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition ${isActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`
            }
          >
            {pg.icono}
            <span>{pg.nombre}</span>
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-gray-200 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-500">Modo:</span>
          <span className={`text-xs font-semibold ${modo === 'live' ? 'text-red-600' : modo === 'testnet' ? 'text-amber-600' : 'text-green-600'}`}>
            {modo.toUpperCase()}
          </span>
        </div>
      </div>
    </aside>
  );
};

