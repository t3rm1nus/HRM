import React from 'react';
import type { FiltrosOrdenes as Filtros, LadoFiltrable } from '../../dominio';

interface FiltrosOrdenesProps {
  filtros: Filtros;
  onCambiar: (cambios: Partial<Pick<Filtros, 'simbolo' | 'lado'>>) => void;
}

const convertirALado = (valor: string): LadoFiltrable | '' =>
  valor === 'buy' || valor === 'sell' ? valor : '';

export const FiltrosOrdenes: React.FC<FiltrosOrdenesProps> = ({ filtros, onCambiar }) => {
  const [simboloEscrito, setSimboloEscrito] = React.useState(filtros.simbolo);

  const aplicarSimbolo = (evento: React.FormEvent<HTMLFormElement>): void => {
    evento.preventDefault();
    onCambiar({ simbolo: simboloEscrito.trim() });
  };

  return (
    <form className="flex flex-wrap items-end gap-3 p-4 border-b border-gray-200" onSubmit={aplicarSimbolo}>
      <label className="flex flex-col text-sm text-gray-600">
        Símbolo
        <input
          type="text"
          value={simboloEscrito}
          onChange={(evento) => setSimboloEscrito(evento.target.value)}
          placeholder="BTCUSDT"
          className="mt-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        />
      </label>
      <label className="flex flex-col text-sm text-gray-600">
        Lado
        <select
          value={filtros.lado}
          onChange={(evento) => onCambiar({ lado: convertirALado(evento.target.value) })}
          className="mt-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        >
          <option value="">Todos</option>
          <option value="buy">Compra</option>
          <option value="sell">Venta</option>
        </select>
      </label>
      <button type="submit" className="px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700">
        Filtrar
      </button>
    </form>
  );
};
