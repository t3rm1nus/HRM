import React, { useMemo } from 'react';
import { usarZonaHoraria } from '../../estado/storeZonaHoraria';
import { ZONA_LOCAL, listarZonas, zonaDelNavegador } from '../../dominio/zonaHoraria';

export const SelectorZonaHoraria: React.FC = () => {
  const zona = usarZonaHoraria((estado) => estado.zona);
  const establecerZona = usarZonaHoraria((estado) => estado.establecerZona);
  const zonas = useMemo(listarZonas, []);

  return (
    <label className="block text-sm font-medium text-gray-700">
      Zona horaria de gráficas y fechas
      <select
        value={zona}
        onChange={(evento) => establecerZona(evento.target.value)}
        className="mt-1 block w-full max-w-sm rounded-lg border border-gray-300 px-3 py-2 text-sm"
      >
        <option value={ZONA_LOCAL}>Automática (navegador: {zonaDelNavegador()})</option>
        {zonas.map((nombre) => <option key={nombre} value={nombre}>{nombre}</option>)}
      </select>
    </label>
  );
};
