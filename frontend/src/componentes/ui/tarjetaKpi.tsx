import React from 'react';

export type Tendencia = 'alza' | 'baja' | 'neutral';

export interface TarjetaKpiProps {
  titulo: string;
  valor: number;
  variacion: number;
  tendencia: Tendencia;
  unidad?: string;
  formato?: (v: number) => string;
  icono?: string;
}

const CONFIG_TENDENCIA: Record<Tendencia, { color: string; icono: string; texto: string }> = {
  alza: { color: 'text-green-600', icono: '↑', texto: 'Alza' },
  baja: { color: 'text-red-600', icono: '↓', texto: 'Baja' },
  neutral: { color: 'text-gray-500', icono: '→', texto: 'Neutral' },
};

export const TarjetaKpi: React.FC<TarjetaKpiProps> = ({
  titulo,
  valor,
  variacion,
  tendencia,
  unidad = '',
  formato,
  icono,
}) => {
  const config = CONFIG_TENDENCIA[tendencia];
  const valorFormateado = formato ? formato(valor) : valor.toLocaleString('es-ES');
  const variacionFormateada = formato ? formato(variacion) : variacion.toLocaleString('es-ES');

  return (
    <div
      className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm"
      role="region"
      aria-label={`KPI: ${titulo}`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-500">{titulo}</span>
        {icono && <span className="text-lg">{icono}</span>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-gray-900">{valorFormateado}</span>
        {unidad && <span className="text-sm text-gray-500">{unidad}</span>}
      </div>
      <div className={`flex items-center gap-1 mt-2 ${config.color}`}>
        <span className="text-sm font-medium">{config.icono}</span>
        <span className="text-sm font-medium">{variacionFormateada}</span>
        <span className="text-xs text-gray-400">{config.texto}</span>
      </div>
    </div>
  );
};

export default TarjetaKpi;
