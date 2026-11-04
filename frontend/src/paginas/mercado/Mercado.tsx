import React, { useEffect, useState } from 'react';
import { GraficaVelas } from '../../componentes/graficas/graficaVelas';
import { usarMercado } from '../../hooks/useMercado';
import { usarZonaHoraria } from '../../estado/storeZonaHoraria';
import { formatearFechaHora, type PreferenciaZona } from '../../dominio/zonaHoraria';
import {
  describirFrescura,
  velaADatosCandle,
  type Frescura,
  type Vela,
} from '../../dominio/mercado';

const CLASES_FRESCURA: Record<Frescura, string> = {
  actual: 'border-emerald-300 bg-emerald-50 text-emerald-900',
  obsoleto: 'border-amber-300 bg-amber-50 text-amber-950',
  sin_datos: 'border-slate-300 bg-slate-50 text-slate-800',
};

const formatearPrecio = (valor: number): string =>
  valor.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const BannerFrescura: React.FC<{ frescura: Frescura; conectado: boolean }> = ({ frescura, conectado }) => (
  <div role="status" className={`rounded-lg border p-3 text-sm ${CLASES_FRESCURA[frescura]}`}>
    <strong>{describirFrescura(frescura)}</strong>
    {frescura === 'obsoleto' && <span> — los precios mostrados no son cotizaciones en directo.</span>}
    {!conectado && <span> · Sin conexión en tiempo real; se actualiza por consulta periódica.</span>}
  </div>
);

const ResumenPrecio: React.FC<{ vela: Vela; obsoleto: boolean; zona: PreferenciaZona }> = ({ vela, obsoleto, zona }) => (
  <p className="text-sm text-slate-700">
    {obsoleto ? 'Último cierre conocido' : 'Último cierre'}:{' '}
    <strong>{formatearPrecio(vela.cierre)} USDT</strong>
    <span className="ml-2 text-xs text-slate-500">{formatearFechaHora(new Date(vela.instante).getTime() / 1000, zona)}</span>
  </p>
);

const SelectorDeSimbolo: React.FC<{
  simbolos: string[];
  seleccionado: string;
  alCambiar: (simbolo: string) => void;
}> = ({ simbolos, seleccionado, alCambiar }) => (
  <label className="block text-sm font-medium text-slate-700">
    Símbolo
    <select
      value={seleccionado}
      onChange={(evento) => alCambiar(evento.target.value)}
      className="ml-2 rounded-lg border border-gray-300 px-3 py-2 text-sm"
    >
      {simbolos.map((simbolo) => <option key={simbolo} value={simbolo}>{simbolo}</option>)}
    </select>
  </label>
);

export const Mercado: React.FC = () => {
  const [simboloElegido, setSimboloElegido] = useState('');
  const { estado, velas, conexion } = usarMercado(simboloElegido);
  const zonaHoraria = usarZonaHoraria((almacen) => almacen.zona);

  const simbolos = estado.data?.simbolos ?? [];
  const primerSimbolo = simbolos[0] ?? '';
  useEffect(() => {
    if (simboloElegido === '' && primerSimbolo !== '') setSimboloElegido(primerSimbolo);
  }, [simboloElegido, primerSimbolo]);

  const frescura = estado.data?.frescura ?? 'sin_datos';
  const lista = velas.data?.velas ?? [];
  const ultima = lista[lista.length - 1];

  return (
    <section className="mx-auto max-w-7xl space-y-4 p-4">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Mercado</h1>
        <p className="mt-1 text-sm text-slate-600">
          Datos públicos de Binance obtenidos por el backend de HRM. Modo PAPER: no se envían órdenes reales.
        </p>
      </header>

      {estado.error !== null && (
        <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          No se pudo consultar el estado del mercado: {estado.error.message}
        </p>
      )}

      {estado.data !== undefined && !estado.data.disponible && (
        <p className="rounded-lg border border-slate-300 bg-slate-50 p-3 text-sm">
          El motor aún se está iniciando; los datos de mercado aparecerán en unos segundos.
        </p>
      )}

      {estado.data !== undefined && <BannerFrescura frescura={frescura} conectado={conexion === 'conectado'} />}

      {simbolos.length > 0 && (
        <SelectorDeSimbolo simbolos={simbolos} seleccionado={simboloElegido} alCambiar={setSimboloElegido} />
      )}

      {velas.error !== null && simboloElegido !== '' && (
        <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          No se pudieron cargar las velas de {simboloElegido}: {velas.error.message}
        </p>
      )}

      {ultima !== undefined && <ResumenPrecio vela={ultima} obsoleto={frescura !== 'actual'} zona={zonaHoraria} />}

      {lista.length > 0 && (
        <GraficaVelas simbolo={simboloElegido} candles={lista.map(velaADatosCandle)} zonaHoraria={zonaHoraria} />
      )}
    </section>
  );
};

export default Mercado;
