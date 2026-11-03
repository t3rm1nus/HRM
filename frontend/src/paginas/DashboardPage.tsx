import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Header, BotónPrimario } from '../componentes';
import { RUTAS } from '../constantes';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-100">
      <Header titulo="Panel de Control" subtitulo="Sistema HRM - Trading" />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 gap-6">
          <section className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Bienvenido al Dashboard</h2>
              <p className="text-sm text-gray-600">
                Esta sección mostrará las métricas principales del sistema. Los datos se cargarán desde la API del backend.
              </p>
              <div className="mt-6 flex flex-wrap gap-4">
                <BotónPrimario tipo="secundario" children={<Link to={RUTAS.DASHBOARD}>Explorar</Link>} />
                <BotónPrimario
                  tipo="success"
                  onClick={() => navigate(RUTAS.DASHBOARD)}
                  children="Actualizar"
                />
                <BotónPrimario
                  tipo="error"
                  onClick={() => navigate('/login')}
                  children="Salir"
                />
              </div>
            </div>
          </section>
          <section className="bg-white p-6 shadow-sm border border-gray-200 rounded-lg">
            <h3 className="text-md font-medium text-gray-900 mb-4">Medidas</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Total Operaciones</p>
                <p className="text-2xl font-bold text-gray-800">0</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Monto Total</p>
                <p className="text-2xl font-bold text-gray-800">S/ 0.00</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Tasa de Éxito</p>
                <p className="text-2xl font-bold text-gray-800">0%</p>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default DashboardPage;