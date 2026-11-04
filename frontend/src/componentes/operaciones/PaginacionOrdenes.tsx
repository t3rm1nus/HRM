import React from 'react';
import { describirRango, type LoteOrdenes } from '../../dominio';

interface PaginacionOrdenesProps {
  lote: LoteOrdenes;
  onCambiarPagina: (pagina: number) => void;
}

const CLASE_BOTON = 'px-3 py-1.5 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50';

export const PaginacionOrdenes: React.FC<PaginacionOrdenesProps> = ({ lote, onCambiarPagina }) => (
  <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
    <span className="text-sm text-gray-500">{describirRango(lote)}</span>
    <div className="flex gap-2">
      <button
        type="button"
        className={CLASE_BOTON}
        disabled={lote.pagina <= 1}
        onClick={() => onCambiarPagina(lote.pagina - 1)}
      >
        Anterior
      </button>
      <button
        type="button"
        className={CLASE_BOTON}
        disabled={!lote.has_more}
        onClick={() => onCambiarPagina(lote.pagina + 1)}
      >
        Siguiente
      </button>
    </div>
  </div>
);
