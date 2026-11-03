import React from 'react';
import { useNavigate } from 'react-router-dom';
import { BotónPrimario, Header } from '../componentes';
import { RUTAS } from '../constantes';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');
  const [esperando, setEsperando] = React.useState(false);

  const manejarEnvío = async (evento: React.FormEvent) => {
    evento.preventDefault();
    setError('');
    setEsperando(true);

    try {
      const api = await import('../api/usuarios');
      await api.fetchHealth();
      navigate(RUTAS.DASHBOARD);
    } catch (err) {
      setError('Credenciales incorrectas. Intente con admin@hrm.local / admin123');
    } finally {
      setEsperando(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <Header titulo="Acceso de Usuario" subtitulo="Sistema HRM - Trading" />
        <div className="bg-white px-6 py-8 shadow-sm border border-gray-200 rounded-xl">
          <h2 className="text-lg font-medium text-gray-900 mb-6">Iniciar Sesión</h2>
          <form className="space-y-4" onSubmit={manejarEnvío}>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                Correo Electrónico
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="admin@hrm.local"
                required
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                Contraseña
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="••••••••"
                required
              />
            </div>
            {error && <p className="text-sm text-red-600">{error}</p>}
            <BotónPrimario
              tipo="primary"
              onClick={() => {}}
              children={esperando ? 'Ingresando...' : 'Ingresar'}
            />
          </form>
          <p className="mt-4 text-sm text-gray-500">
            Credenciales de prueba:
            <br />
            <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">admin@hrm.local</code>
            <br />
            <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">admin123</code>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;