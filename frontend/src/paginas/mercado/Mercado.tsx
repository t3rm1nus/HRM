import React, { useState, useRef } from 'react';
import { useModoStore } from '../estado/storeModo';
import { SelectorSimbolo } from '../componentes/mercado/SelectorSimbolo';
import { GraficaMercado } from '../componentes/mercado/GraficaMercado';
import { PanelRSI } from '../componentes/mercado/PanelRSI';
import { PanelMACD } from '../componentes/mercado/PanelMACD';
import { PanelMultitemporal } from '../componentes/mercado/PanelMultitemporal';
import { ComparadorNormalizado } from '../componentes/mercado/ComparadorNormalizado';
import { usePaginas } from '../constantes/paginas';

export const Mercado: React.FC = () => {
  const { modo } = useModoStore();
  const paginas = usePaginas();

  // Estado del selector
  const [simbolo, setSimbolo] = useState('BTC/USD');
  const [timeframe, setTimeframe] = useState('1m');

  // Datos simulados - en producción vendrían del WebSocket/MSW
  const [velas, setVelas] = useState<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[]>([]);

  const [datosCierresBTC, setDatosCierresBTC] = useState<{
    timestamp: string;
    cierre: number;
  }[]>([]);

  const [datosCierresETH, setDatosCierresETH] = useState<{
    timestamp: string;
    cierre: number;
  }[]>([]);

  // Simular datos cada vez que cambia el timeframe
  useEffect(() => {
    const simularDatos = () => {
      const nuevasVelas = [];
      const ahora = new Date();
      for (let i = 0; i < 100; i++) {
        const fecha = new Date(ahora.getTime() - i * 60000); // 1 minuto intervals
        const basePrecio = simbolo === 'BTC/USD' ? 67000 : 3800;
        const variacion = (Math.random() - 0.5) * 200;
        const precio = Math.max(basePrecio + variacion, 10);
        nuevasVelas.push({
          timestamp: fecha.toISOString(),
          open: precio - 10,
          high: precio + 10,
          low: precio - 10,
          close: precio,
          volume: Math.random() * 1000000,
        });
      }
      setVelas(nuevasVelas);

      // Simular cierres BTC/ETH
      const nuevosCierresBTC = nuevasVelas.map((v, i) => ({
        timestamp: v.timestamp,
        cierre: v.close,
      }));
      const nuevosCierresETH = nuevasVelas.map((v, i) => ({
        timestamp: v.timestamp,
        cierre: v.close * 0.058, // Relación simplificada ETH/BTC
      }));
      setDatosCierresBTC(nuevosCierresBTC);
      setDatosCierresETH(nuevosCierresETH);
    };

    simularDatos();
    const interval = setInterval(simularDatos, 1000); // Actualizar cada segundo

    return () => clearInterval(interval);
  }, [simbolo, timeframe]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Mercado</h1>

      {/* Selector de símbolo y timeframe */}
      <div className="border rounded-lg p-4 mb-6 bg-white shadow-sm">
        <SelectorSimbolo
          simbolos={['BTC/USD', 'ETH/USD', 'SOL/USD']}
          simboloSeleccionado={simbolo}
          onSimboloCambiar={setSimbolo}
          timeframes={'1m', '5m', '1h'}
          timeframeSeleccionado={timeframe}
          onTimeframeCambiar={setTimeframe}
        />
      </div>

      {/* Panel superior:Comparador Normalizado */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <ComparadorNormalizado
          btcDatos={datosCierresBTC}
          ethDatos={datosCierresETH}
        />
      </div>

      {/* Panel multitemporal */}
      <PanelMultitemporal
        simbolo={simbolo}
        timeframes={'1m', '5m', '1h'}
        setTimeframe={setTimeframe}
      />

      {/* Gráfica principal */}
      <GraficaMercado
        simbolo={simbolo}
        velas={velas}
        timeframe={timeframe}
        capas={{
          ma50: timeframe !== '1m',
          ma200: timeframe !== '1m' && timeframe !== '5m',
          bollinger: true,
          soportes: true,
          resistencias: true,
        }}
      />

      {/* Paneles laterales sincronizados */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <PanelRSI
          datosCierres={datosCierresBTC}
          timeframe={timeframe}
        />
        <PanelMACD
          datosCierres={datosCierresBTC}
          timeframe={timeframe}
        />
      </div>
    </div>
  );
};

export default Mercado;