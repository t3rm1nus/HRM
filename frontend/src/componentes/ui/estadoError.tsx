import React from 'react';

export interface EstadoErrorProps {
  mensaje: string;
  detalle?: string;
  onReintentar?: () => void;
  onCerrar?: () => void;
}

export const EstadoError: React.FC<EstadoErrorProps> = ({
  mensaje,
  detalle,
  onReintentar,
  onCerrar,
}) => {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 py-8"
      role="alert"
      aria-live="assertive"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
        <span className="text-2xl text-red-600">!</span>
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-red-700">{mensaje}</p>
        {detalle && <p className="mt-1 text-xs text-gray-500">{detalle}</p>}
      </div>
      <div className="flex gap-2">
        {onReintentar && (
          <button
            type="button"
            className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700"
            onClick={onReintentar}
          >
            Reintentar
          </button>
        )}
        {onCerrar && (
          <button
            type="button"
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300"
            onClick={onCerrar}
          >
            Cerrar
          </button>
        )}
      </div>
    </div>
  );
};

export default EstadoError;
