import React from 'react';

export interface EstadoCargaProps {
  mensaje?: string;
  tamaño?: 'small' | 'medium' | 'large';
  animado: boolean;
}

const TAMANOS = {
  small: 'h-4 w-4',
  medium: 'h-8 w-8',
  large: 'h-12 w-12',
} as const;

export const EstadoCarga: React.FC<EstadoCargaProps> = ({
  mensaje = 'Cargando...',
  tamaño = 'medium',
  animado = true,
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 py-8 ${animado ? 'animate-pulse' : ''}`}
      role="status"
      aria-label={mensaje}
    >
      <div className={`${TAMANOS[tamaño]} rounded-full border-2 border-blue-500 border-t-transparent origin-center`} />
      <p className="text-sm text-gray-500">{mensaje}</p>
    </div>
  );
};

export default EstadoCarga;
