import React from 'react';

export interface PanelLateralDetalleProps {
  abierto: boolean;
  onCerrar: () => void;
  titulo: string;
  children: React.ReactNode;
  ancho?: 'sm' | 'md' | 'lg';
}

const ANCHOS = {
  sm: 'w-80',
  md: 'w-96',
  lg: 'w-[32rem]',
} as const;

export const PanelLateralDetalle: React.FC<PanelLateralDetalleProps> = ({
  abierto,
  onCerrar,
  titulo,
  children,
  ancho = 'md',
}) => {
  React.useEffect(() => {
    if (abierto) {
      const handleEsc = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          onCerrar();
        }
      };
      document.addEventListener('keydown', handleEsc);
      return () => document.removeEventListener('keydown', handleEsc);
    }
  }, [abierto, onCerrar]);

  if (!abierto) {
    return null;
  }

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-black/40"
        onClick={onCerrar}
        aria-hidden="true"
      />
      <aside
        className={`fixed right-0 top-0 z-50 h-full ${ANCHOS[ancho]} bg-white shadow-2xl transition-transform duration-300`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="panel-titulo"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
          <h2 id="panel-titulo" className="text-lg font-semibold text-gray-900">
            {titulo}
          </h2>
          <button
            type="button"
            className="rounded-lg p-1 hover:bg-gray-100"
            onClick={onCerrar}
            aria-label="Cerrar panel"
          >
            ✕
          </button>
        </div>
        <div className="overflow-y-auto p-6">{children}</div>
      </aside>
    </>
  );
};

export default PanelLateralDetalle;
