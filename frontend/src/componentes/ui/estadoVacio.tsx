import React from 'react';

export interface EstadoVacioProps {
  mensaje: string;
  detalle?: string;
  accion?: {
    texto: string;
    onClick: () => void;
  };
  icono?: string;
}

export const EstadoVacio: React.FC<EstadoVacioProps> = ({
  mensaje,
  detalle,
  accion,
  icono = '📋',
}) => {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 py-8"
      role="status"
      aria-live="polite"
    >
      <div className="text-4xl" aria-hidden="true">
        {icono}
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-gray-700">{mensaje}</p>
        {detalle && <p className="mt-1 text-xs text-gray-500">{detalle}</p>}
      </div>
      {accion && (
        <button
          type="button"
          className="mt-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
          onClick={accion.onClick}
        >
          {accion.texto}
        </button>
      )}
    </div>
  );
};

export default EstadoVacio;
