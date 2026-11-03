import React from 'react';

export type SemaforoEstado = 'verde' | 'amarillo' | 'rojo' | 'gris';

export interface SemaforoProps {
  estado: SemaforoEstado;
  tamaño?: 'small' | 'medium' | 'large';
  etiqueta?: string;
  animado?: boolean;
}

const COLORES: Record<SemaforoEstado, string> = {
  verde: 'bg-green-500',
  amarillo: 'bg-amber-500',
  rojo: 'bg-red-500',
  gris: 'bg-gray-400',
};

const TAMANOS = {
  small: 'w-3 h-3',
  medium: 'w-4 h-4',
  large: 'w-6 h-6',
} as const;

export const Semaforo: React.FC<SemaforoProps> = ({
  estado,
  tamaño = 'medium',
  etiqueta,
  animado = false,
}) => {
  return (
    <div
      className="inline-flex items-center gap-2"
      role="status"
      aria-label={etiqueta || `Estado: ${estado}`}
    >
      <div
        className={`rounded-full ${COLORES[estado]} ${TAMANOS[tamaño]} ${
          animado ? 'animate-pulse' : ''
        }`}
      />
      {etiqueta && (
        <span className="text-sm font-medium text-gray-700">{etiqueta}</span>
      )}
    </div>
  );
};

export default Semaforo;
