import React from 'react';
import { EstadoCarga, EstadoError, EstadoVacio } from '../../componentes/ui';
import { FiltrosOrdenes, PaginacionOrdenes, TablaOrdenes } from '../../componentes/operaciones';
import {
  FILTROS_ORDENES_INICIALES,
  cambiarFiltros,
  irAPagina,
  type FiltrosOrdenes as Filtros,
  type LoteOrdenes,
} from '../../dominio';
import { usarOperaciones } from '../../hooks/useOperacionesQuery';

export const MENSAJE_OPERACIONES_SIN_DATOS = 'No hay operaciones que mostrar.';
export const DETALLE_OPERACIONES_SIN_DATOS =
  'Aquí aparecerán las órdenes ejecutadas, o ninguna si los filtros no coinciden con nada.';
export const MENSAJE_OPERACIONES_ERROR = 'No se pudieron obtener las operaciones';

const ContenedorPagina: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="max-w-7xl mx-auto px-4 py-8">
    <h1 className="text-2xl font-bold text-gray-900">Operaciones</h1>
    <div className="mt-6 bg-white border border-gray-200 rounded-lg">{children}</div>
  </div>
);

interface ContenidoOperacionesProps {
  lote: LoteOrdenes | undefined;
  error: Error | null;
  cargando: boolean;
  onReintentar: () => void;
  onCambiarPagina: (pagina: number) => void;
}

const ContenidoOperaciones: React.FC<ContenidoOperacionesProps> = ({
  lote,
  error,
  cargando,
  onReintentar,
  onCambiarPagina,
}) => {
  if (cargando) return <EstadoCarga animado mensaje="Cargando operaciones..." />;
  if (error) {
    return <EstadoError mensaje={MENSAJE_OPERACIONES_ERROR} detalle={error.message} onReintentar={onReintentar} />;
  }
  if (!lote || lote.total === 0) {
    return <EstadoVacio mensaje={MENSAJE_OPERACIONES_SIN_DATOS} detalle={DETALLE_OPERACIONES_SIN_DATOS} />;
  }
  return (
    <>
      <TablaOrdenes ordenes={lote.ordenes} />
      <PaginacionOrdenes lote={lote} onCambiarPagina={onCambiarPagina} />
    </>
  );
};

export const PaginaOperaciones: React.FC = () => {
  const [filtros, setFiltros] = React.useState<Filtros>(FILTROS_ORDENES_INICIALES);
  const { datos, error, isLoading, refetch } = usarOperaciones(filtros);

  return (
    <ContenedorPagina>
      <FiltrosOrdenes filtros={filtros} onCambiar={(cambios) => setFiltros(cambiarFiltros(filtros, cambios))} />
      <ContenidoOperaciones
        lote={datos}
        error={error}
        cargando={isLoading}
        onReintentar={refetch}
        onCambiarPagina={(pagina) => setFiltros(irAPagina(filtros, pagina))}
      />
    </ContenedorPagina>
  );
};
