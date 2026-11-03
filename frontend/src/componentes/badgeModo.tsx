import React from 'react';
import { useModoStore } from '../estado/storeModo';
import { MODOS_CONFIG } from '../dominio/modo';

export const BadgeModo: React.FC = () => {
  const { modo } = useModoStore();
  const config = MODOS_CONFIG[modo];

  return (
    <span
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wide"
      style={{ color: config.color, backgroundColor: config.fondo }}
      title={`Modo: ${config.etiqueta}`}
    >
      <span
        className="inline-block w-2.5 h-2.5 rounded-full"
        style={{ backgroundColor: config.color }}
      />
      {config.etiqueta}
    </span>
  );
};
