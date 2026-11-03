import React from 'react';
import { BadgeModo } from '../../componentes/ui/badgeModo';
import { InsigniaOrigen } from '../../componentes/ui/insigniaOrigen';
import { TarjetaKpi } from '../../componentes/ui/tarjetaKpi';
import { useModoStore } from '../../estado/storeModo';

export const KPIsPrincipales: React.FC = () => {
  const modoActual = useModoStore((estado) => estado.modo);
  // Datos simulados para KPIs
  const navTotal = 12547.89;
  const pnlTotalDiario = 285.50;
  const drawdownActual = -12.50;
  const drawdownMaximo = -45.80;
  const winRate = 67.5;
  const cicloActual = 12;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm">
      <h2 className="text-lg font-medium text-gray-600 mb-6">KPIs Principales</h2>

      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* NAV */}
        <TarjetaKpi
          titulo="NAV Total"
          valor={navTotal}
          variacion={2.3}
          tendencia="alza"
          unidad="USD"
          icono="💰"
        />
        {/* PnL Diario */}
        <TarjetaKpi
          titulo="PnL Diario"
          valor={pnlTotalDiario}
          variacion={2.3}
          tendencia="alza"
          unidad="USD"
          icono="📈"
        />
        {/* Drawdown Actual */}
        <TarjetaKpi
          titulo="Drawdown Actual"
          valor={drawdownActual}
          variacion={drawdownActual - drawdownMaximo < 0 ? 31.3 : -31.3}
          tendencia="baja"
          unidad="%"
          icono="📉"
        />
        {/* Drawdown Máximo */}
        <TarjetaKpi
          titulo="Drawdown Máx."
          valor={drawdownMaximo}
          variacion={0}
          tendencia="baja"
          unidad="%"
          icono="📉"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Win Rate */}
        <TarjetaKpi
          titulo="Win Rate"
          valor={winRate}
          variacion={winRate - 65}
          tendencia="alza"
          unidad="%"
          icono="✅"
        />
        {/* Ciclo */}
        <TarjetaKpi
          titulo="Ciclo Actual"
          valor={cicloActual}
          variacion={1}
          tendencia="alza"
          unidad="#"
          icono="🔄"
        />
      </div>

      {/* Insignias de modo y origen */}
      <div className="mt-4 flex items-center gap-3">
        <BadgeModo modo={modoActual === "papel" ? "paper" : modoActual} />
        <InsigniaOrigen
          tipo="sintetico"
          origen="API"
          antiguedad="2h"
          descripcion="Datos de mercado"
        />
        <InsigniaOrigen
          tipo="respaldo"
          origen="Base datos"
          antiguedad="1d"
          descripcion="Historial trade"
        />
      </div>
    </div>
  );
};

export default KPIsPrincipales;