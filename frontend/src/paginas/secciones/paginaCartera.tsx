import React from 'react';
import { calcularRentabilidad } from '../../dominio/cartera';
import { EstadoCarga, EstadoError, EstadoVacio } from '../../componentes/ui';
import { usarPortafolio, type PortfolioRespuesta } from '../../hooks/usePortafolioQuery';

const formatearUsdt = (valor: number): string =>
  `${valor.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT`;

const formatearCantidad = (valor: number): string =>
  valor.toLocaleString('es-ES', { minimumFractionDigits: 0, maximumFractionDigits: 6 });

const formatearPorcentaje = (valor: number | null): string =>
  valor === null
    ? 'no disponible'
    : `${valor.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} %`;

const obtenerColorPnl = (pnl: number): string => (pnl >= 0 ? 'text-green-600' : 'text-red-600');

const ContenedorPagina: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="max-w-7xl mx-auto px-4 py-8">
    <h1 className="text-2xl font-bold text-gray-900">Cartera</h1>
    {children}
  </div>
);

const DatoResumen: React.FC<{ titulo: string; valor: string; clase?: string }> = ({ titulo, valor, clase = 'text-gray-900' }) => (
  <div className="bg-white border rounded-xl p-4">
    <p className="text-sm text-gray-500">{titulo}</p>
    <p className={`mt-1 text-xl font-bold ${clase}`}>{valor}</p>
  </div>
);

const ResumenCartera: React.FC<{ datos: PortfolioRespuesta }> = ({ datos }) => {
  const { pnl, porcentaje } = calcularRentabilidad(datos.total_nav, datos.initial_capital);
  const clasePnl = obtenerColorPnl(pnl);

  return (
    <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
      <DatoResumen titulo="NAV total" valor={formatearUsdt(datos.total_nav)} />
      <DatoResumen titulo="Capital inicial" valor={formatearUsdt(datos.initial_capital)} />
      <DatoResumen titulo="PnL" valor={formatearUsdt(pnl)} clase={clasePnl} />
      <DatoResumen titulo="Rentabilidad" valor={formatearPorcentaje(porcentaje)} clase={clasePnl} />
    </div>
  );
};

const BalancesCartera: React.FC<{ datos: PortfolioRespuesta }> = ({ datos }) => (
  <div className="mt-6 bg-white border rounded-xl p-4">
    <h2 className="font-semibold mb-2">Balances</h2>
    <table className="min-w-full text-sm">
      <thead>
        <tr>
          <th className="text-left">Activo</th>
          <th className="text-left">Cantidad</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>BTC</td><td>{formatearCantidad(datos.btc_balance)}</td></tr>
        <tr><td>ETH</td><td>{formatearCantidad(datos.eth_balance)}</td></tr>
        <tr><td>USDT</td><td>{formatearCantidad(datos.usdt_balance)}</td></tr>
      </tbody>
    </table>
  </div>
);

const SeccionesSinFuente: React.FC = () => (
  <div className="mt-6 bg-white border rounded-xl p-4">
    <EstadoVacio
      icono="🔌"
      mensaje="Asignación, lotes FIFO y rebalanceos no conectados"
      detalle="El backend aún no publica estos datos. No se muestran valores de ejemplo."
    />
  </div>
);

export const PaginaCartera: React.FC = () => {
  const { datos, error, isLoading, refetch } = usarPortafolio();

  if (isLoading) {
    return <ContenedorPagina><EstadoCarga mensaje="Cargando cartera…" animado /></ContenedorPagina>;
  }
  if (error) {
    return (
      <ContenedorPagina>
        <EstadoError mensaje="No se pudo obtener la cartera" detalle={error.message} onReintentar={refetch} />
      </ContenedorPagina>
    );
  }
  if (!datos) {
    return <ContenedorPagina><EstadoVacio mensaje="Sin datos de cartera" /></ContenedorPagina>;
  }

  return (
    <ContenedorPagina>
      <ResumenCartera datos={datos} />
      <BalancesCartera datos={datos} />
      <SeccionesSinFuente />
    </ContenedorPagina>
  );
};
