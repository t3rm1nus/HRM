import React from 'react';

export type TipoOrigen = 'sintetico' | 'respaldo' | 'obsoleto';

export interface InsigniaOrigenProps {
  tipo: TipoOrigen;
  origen: string;
  antiguedad: string;
  descripcion?: string;
}

const CONFIG: Record<TipoOrigen, { texto: string; color: string; fondo: string; borde: string }> = {
  sintetico: {
    texto: 'SINTÉTICO',
    color: 'text-amber-700',
    fondo: 'bg-amber-50',
    borde: 'border-amber-200',
  },
  respaldo: {
    texto: 'RESPALDO',
    color: 'text-blue-700',
    fondo: 'bg-blue-50',
    borde: 'border-blue-200',
  },
  obsoleto: {
    texto: 'OBSOLETO',
    color: 'text-red-700',
    fondo: 'bg-red-50',
    borde: 'border-red-200',
  },
};

export const InsigniaOrigen: React.FC<InsigniaOrigenProps> = ({
  tipo,
  origen,
  antiguedad,
  descripcion,
}) => {
  const config = CONFIG[tipo];
  const [mostrarTooltip, setMostrarTooltip] = React.useState(false);

  return (
    <div className="relative inline-block">
      <button
        type="button"
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wide border ${config.color} ${config.fondo} ${config.borde} cursor-help`}
        onMouseEnter={() => setMostrarTooltip(true)}
        onMouseLeave={() => setMostrarTooltip(false)}
        onFocus={() => setMostrarTooltip(true)}
        onBlur={() => setMostrarTooltip(false)}
        aria-label={`Origen: ${origen}, antigüedad: ${antiguedad}`}
      >
        <span className={`inline-block w-2 h-2 rounded-full ${config.color.replace('text-', 'bg-')}`} />
        {config.texto}
      </button>
      {mostrarTooltip && (
        <div
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-lg bg-gray-900 text-white text-xs p-3 shadow-lg"
          role="tooltip"
        >
          <div className="font-semibold mb-1">{config.texto}</div>
          <div className="text-gray-300">Origen: {origen}</div>
          <div className="text-gray-300">Antigüedad: {antiguedad}</div>
          {descripcion && <div className="text-gray-300 mt-1">{descripcion}</div>}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-gray-900" />
        </div>
      )}
    </div>
  );
};

export default InsigniaOrigen;
