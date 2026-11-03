import React, { useEffect, useRef } from 'react';
import { useModoStore } from '../estado/storeModo';
import { BadgeModo } from '../componentes/badgeModo';
import { Semaforo } from '../componentes/ui/semaforo';
import { InsigniaOrigen } from '../componentes/ui/insigniaOrigen';
import { TarjetaKpi } from '../componentes/ui/tarjetaKpi';
import { KPIsPrincipales } from './KPIsPrincipales';
import { CurvaEquity } from './CurvaEquity';
import { PosicionesAbiertas } from './PosicionesAbiertas';
import { SenalesOrdenes } from './SenalesOrdenes';
import { SemáforoSalud } from './SemáforoSalud';
import { usePaginas } from '../constantes/paginas';

export const Panel: React.FC = () => {
  const { modo } = useModoStore();
  const paginas = usePaginas();

  // Simular datos WebSocket (en producción vendrían del WebSocket)
  const [ultimaActualizacion, setUltimaActualizacion] = useState<Date | null>(null);

  useEffect(() => {
    // Simular conexión WebSocket cada 5 segundos
    const interval = setInterval(() => {
      setUltimaActualizacion(new Date());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Panel de Control</h1>

      {/* Timestamp última actualización */}
      {ultimaActualizacion && (
        <p className="text-sm text-gray-500 mb-6">
          Última actualización: {ultimaActualizacion.toLocaleTimeString()}
        </p>
      )

      {/* KPIs Principales */}
      <KPIsPrincipales />

      {/* Curva de Equity */}
      <CurvaEquity
        datosOperaciones={[
          { timestamp: 'hace 1h', pnl: 50, regimen: 'LIVE', tipo: 'ingreso' },
          { timestamp: 'hace 50min', pnl: -20, regimen: 'LIVE', tipo: 'egreso' },
          { timestamp: 'hace 40min', pnl: 35, regimen: 'LIVE', tipo: 'ingreso' },
          { timestamp: 'hace 30min', pnl: -10, regimen: 'LIVE', tipo: 'egreso' },
          { timestamp: 'hace 20min', pnl: 75, regimen: 'LIVE', tipo: 'ingreso' },
          { timestamp: 'hace 10min', pnl: -15, regimen: 'LIVE', tipo: 'egreso' },
        ]}
      />

      {/* Posiciones Abiertas */}
      <PosicionesAbiertas />

      {/* Señales y Órdenes */}
      <SenalesOrdenes />

      {/* Semáforo de Salud y Régimen L3 */}
      <div className="grid grid-cols-2 gap-4 my-8">
        <SemáforoSalud />
        {/* RegimenL3 - placeholder, se integraría con store real */}
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <h3 className="text-lg font-medium text-gray-600">Régimen L3</h3>
          <p className="text-sm text-gray-500">
            Régimen L3 activo - Modo: <span className="font-semibold text-green-600">{modo.toUpperCase()}</span>
          </p>
          <InsigniaOrigen
            tipo="sintetico"
            origen="API"
            antiguedad="1h"
            descripcion="Datos de mercado L3"
          />
          <InsigniaOrigen
            tipo="respaldo"
            origen="Base datos"
            antiguedad="6h"
            descripcion="Historial L3"
          />
        </div>
      </div>

      {/* Grid de páginas */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {paginas.map((pg) => (
          <div key={pg.ruta} className="bg-white rounded-lg p-4 shadow-sm">
            <h3 className="text-sm font-medium text-gray-600 uppercase tracking-wider mb-3">{pg.nombre}</h3>
            <p className="text-sm text-gray-500">Contenido de {pg.nombre}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Panel;