import React from 'react';
import { useModoStore, ModoAplicacion } from '../estado/storeModo';

export const ThemeToggle: React.FC = () => {
  const { modo, establecerModo } = useModoStore();

  const handleCambiarModo = () => {
    if (modo === ModoAplicacion.LIVE) {
      establecerModo(ModoAplicacion.PAPERA);
    } else if (modo === ModoAplicacion.TESTNET) {
      establecerModo(ModoAplicacion.LIVE);
    } else {
      establecerModo(ModoAplicacion.TESTNET);
    }
  };

  const iconoModo: Record<ModoAplicacion, string> = {
    [ModoAplicacion.PAPERA]: '🟢',
    [ModoAplicacion.TESTNET]: '🟡',
    [ModoAplicacion.LIVE]: '🔴',
  };

  return (
    <button
      type="button"
      className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 text-sm font-medium text-gray-700 hover:bg-gray-50"
      onClick={handleCambiarModo}
      title="Cambiar modo"
    >
      {iconoModo[modo]}
      <span>{modo.toUpperCase()}</span>
    </button>
  );
};

