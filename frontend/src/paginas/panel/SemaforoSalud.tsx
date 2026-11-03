import React from 'react';
import { Semaforo } from '../../componentes/ui/semaforo';

export const SemaforoSalud: React.FC = () => {
  return (
    <div>
      <h2 className="text-lg font-medium text-gray-600 mb-3">Salud del Sistema</h2>
      <div className="grid grid-cols-2 gap-4">
        <Semaforo estado="verde" etiqueta="Conexión WS" />
        <Semaforo estado="verde" etiqueta="Datos Válidos" />
        <Semaforo estado="amarillo" etiqueta="Latencia Alta" />
        <Semaforo estado="rojo" etiqueta="Alertas Activas" />
      </div>

      <div className="mt-4 p-3 border rounded-lg bg-gray-50">
        <h3 className="text-sm font-medium text-gray-600 mb-2">Alertas Activas</h3>
        <ul className="text-sm text-gray-700 space-y-1">
          <li>⚠️ Latencia superior a 2s detectada</li>
          <li>⚠️ Drawdown acercándose al límite</li>
          <li>⚠️ Posición sobreapalancada</li>
        </ul>
      </div>
    </div>
  );
};

export default SemaforoSalud;