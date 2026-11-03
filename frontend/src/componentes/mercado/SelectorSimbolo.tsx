import React, { useState } from 'react';

export interface SelectorSimboloProps {
  simbolos: string[];
  simboloSeleccionado: string;
  onSimboloCambiar: (simbolo: string) => void;
  timeframes: string[];
  timeframeSeleccionado: string;
  onTimeframeCambiar: (timeframe: string) => void;
}

export const SelectorSimbolo: React.FC<SelectorSimboloProps> = ({
  simbolos,
  simboloSeleccionado,
  onSimboloCambiar,
  timeframes,
  timeframeSeleccionado,
  onTimeframeCambiar,
}) => {
  return (
    <div className="border rounded-lg p-3 mb-4 flex flex-col sm:flex-row gap-2">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Símbolo</label>
        <select
          value={simboloSeleccionado}
          onChange={(e) => onSimboloCambiar(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-200"
        >
          {simbolos.map((simbolo) => (
            <option key={simbolo} value={simbolo}>
              {simbolo}
            </option>
          ))}
        </option>
      </select>
    </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
        <select
          value={timeframeSeleccionado}
          onChange={(e) => onTimeframeCambiar(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-200"
        >
          {timeframes.map((timeframe) => (
            <option key={timeframe} value={timeframe}>
              {timeframe}
            </option>
          ))}
        </option>
      </select>
    </div>
  </div>
};

export default SelectorSimbolo;