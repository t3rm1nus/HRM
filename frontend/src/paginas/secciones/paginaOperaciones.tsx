import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { TablaDatos } from '../../componentes/ui/tablaDatos';
import { PanelLateralDetalle } from '../../componentes/ui/panelLateralDetalle';
import { TRADES_FIQUETAS } from '../../mocks/fixtures';

interface Operacion {
  id: string;
  simbodo: string;
  lado: 'buy' | 'sell';
  fecha: string;
  estrategia: string;
  estado: 'ejecutada' | 'rechazada' | 'bloqueada';
  motivo: string;
  condicionesMercado: string;
  senalOrigen: string;
  ciclo: number;
  pnl: number;
  comision: number;
}

const ESTRATEGIAS = ['momentum', 'mean_reversion', 'breakout'] as const;
const ESTADOS_TAB = ['ejecutada', 'rechazada', 'bloqueada'] as const;

const construirOperaciones = (): Operacion[] =>
  TRADES_FIQUETAS.map((t, i) => ({
    id: t.operation_id,
    simbodo: t.symbol,
    lado: t.side,
    fecha: t.timestamp.slice(0, 10),
    estrategia: ESTRATEGIAS[i % ESTRATEGIAS.length] ?? ESTRATEGIAS[0],
    estado: ESTADOS_TAB[i % ESTADOS_TAB.length] ?? ESTADOS_TAB[0],
    motivo:
      i % 3 === 1
        ? 'Confianza por debajo del umbral'
        : i % 3 === 2
        ? 'Cooldown activo tras operacion reciente'
        : 'Ejecutada correctamente',
    condicionesMercado: `RSI ${40 + (i % 20)}, volatilidad ${(0.01 + (i % 5) * 0.005).toFixed(3)}`,
    senalOrigen: `L2-señal-${i + 1}`,
    ciclo: 100 + i,
    pnl: (i % 2 === 0 ? 1 : -1) * (t.net_value * 0.02),
    comision: t.commission,
  }));

const OPERACIONES = construirOperaciones();

export const PaginaOperaciones: React.FC = () => {
  const [params, setParams] = useSearchParams();
  const [detalle, setDetalle] = React.useState<Operacion | null>(null);

  const simbodo = params.get('simbodo') ?? '';
  const lado = params.get('lado') ?? '';
  const fecha = params.get('fecha') ?? '';
  const estrategia = params.get('estrategia') ?? '';
  const tab = (params.get('tab') ?? 'ejecutada') as Operacion['estado'];

  const actualizar = (clave: string, valor: string) => {
    const next = new URLSearchParams(params);
    if (valor) next.set(clave, valor);
    else next.delete(clave);
    setParams(next);
  };

  const filtradas = OPERACIONES.filter((o) => {
    if (o.estado !== tab) return false;
    if (simbodo && o.simbodo !== simbodo) return false;
    if (lado && o.lado !== lado) return false;
    if (fecha && o.fecha !== fecha) return false;
    if (estrategia && o.estrategia !== estrategia) return false;
    return true;
  });

  const exportarCSV = () => {
    const csv = [
      ['id', 'simbodo', 'lado', 'fecha', 'estrategia', 'estado', 'pnl', 'comision'].join(','),
      ...filtradas.map((o) =>
        [o.id, o.simbodo, o.lado, o.fecha, o.estrategia, o.estado, o.pnl, o.comision].join(',')
      ),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'operaciones.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const columnas = [
    { clave: 'simbodo', titulo: 'Simbodo', sortable: true },
    { clave: 'lado', titulo: 'Lado', sortable: true },
    { clave: 'fecha', titulo: 'Fecha', sortable: true },
    { clave: 'estrategia', titulo: 'Estrategia', sortable: true },
    { clave: 'pnl', titulo: 'PnL', sortable: true, render: (v: number) => v.toFixed(2) },
    {
      clave: 'id',
      titulo: 'Acción',
      render: (_: any, fila: Operacion) => (
        <button
          type="button"
          className="text-blue-600 hover:underline"
          onClick={() => setDetalle(fila)}
        >
          Ver detalle
        </button>
      ),
    },
  ];

  const simbodos = Array.from(new Set(OPERACIONES.map((o) => o.simbodo)));

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Operaciones</h1>
      <div className="mt-4 flex gap-2">
        {ESTADOS_TAB.map((e) => (
          <button
            key={e}
            type="button"
            onClick={() => actualizar('tab', e)}
            className={`px-3 py-1.5 text-sm rounded-lg ${tab === e ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
          >
            {e}
          </button>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-5 gap-2">
        <select value={simbodo} onChange={(e) => actualizar('simbodo', e.target.value)} className="px-3 py-2 border rounded-lg text-sm">
          <option value="">Simbodo</option>
          {simbodos.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select value={lado} onChange={(e) => actualizar('lado', e.target.value)} className="px-3 py-2 border rounded-lg text-sm">
          <option value="">Lado</option>
          <option value="buy">buy</option>
          <option value="sell">sell</option>
        </select>
        <input type="date" value={fecha} onChange={(e) => actualizar('fecha', e.target.value)} className="px-3 py-2 border rounded-lg text-sm" />
        <select value={estrategia} onChange={(e) => actualizar('estrategia', e.target.value)} className="px-3 py-2 border rounded-lg text-sm">
          <option value="">Estrategia</option>
          {ESTRATEGIAS.map((e) => (
            <option key={e} value={e}>
              {e}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => {
            const next = new URLSearchParams();
            next.set('tab', tab);
            setParams(next);
          }}
          className="px-3 py-2 text-sm bg-gray-100 rounded-lg"
        >
          Limpiar
        </button>
      </div>
      <div className="mt-6">
        <TablaDatos datos={filtradas} columnas={columnas} exportable onExportar={exportarCSV} />
      </div>
      <PanelLateralDetalle abierto={!!detalle} onCerrar={() => setDetalle(null)} titulo="Detalle de operación">
        {detalle && (
          <div className="space-y-2 text-sm">
            <p><strong>Motivo:</strong> {detalle.motivo}</p>
            <p><strong>Condiciones de mercado:</strong> {detalle.condicionesMercado}</p>
            <p><strong>Señal origen:</strong> {detalle.senalOrigen}</p>
            <p><strong>Ciclo:</strong> {detalle.ciclo}</p>
            <p><strong>PnL:</strong> {detalle.pnl.toFixed(2)}</p>
            <p><strong>Comisión:</strong> {detalle.comision}</p>
          </div>
        )}
      </PanelLateralDetalle>
    </div>
  );
};
