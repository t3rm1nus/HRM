import React from 'react';
import { Link } from 'react-router-dom';

export const ErrorGlobalPage: React.FC<{ error: string }> = ({ error }) => {
  return (
    <div className="flex h-full min-h-screen items-center justify-center bg-gray-100">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-red-600">500</h1>
        <p className="mt-4 text-xl text-gray-600">Error interno del servidor</p>
        <p className="mt-2 text-gray-500">{error || 'Ocurrió un error inesperado.'}</p>
        <div className="mt-6 flex justify-center gap-4">
          <Link to="/" className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Volver al inicio
          </Link>
          <Link to="/dashboard" className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
            Explorar
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ErrorGlobalPage;
