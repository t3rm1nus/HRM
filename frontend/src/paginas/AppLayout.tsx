import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Header, BotónPrimario } from '../componentes';
import { RUTAS } from '../constantes';

const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header titulo="Sistema HRM" subtitulo="Trading" />
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-4">
            <NavLink
              to={RUTAS.INICIO}
              className={({ isActive }) => `px-3 py-2 rounded-md text-sm font-medium ${isActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`
              }
            >
              Inicio
            </NavLink>
            <NavLink
              to={RUTAS.INICIO}
              className={({ isActive }) => `px-3 py-2 rounded-md text-sm font-medium ${isActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'}`
              }
            >
              Dashboard
            </NavLink>
            <BotónPrimario tipo="success" children="Cerrar Sesión" onClick={() => {}} />
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
};

export default AppLayout;