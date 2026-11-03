import { useMemo } from 'react';
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { calcularMacd } from '../../dominio/indicadores/calculos';

export interface PanelMACDProps {
  datosCierres: { timestamp: string; cierre: number }[];
  timeframe: string;
  tema?: 'light' | 'dark';
}

export const PanelMACD = ({ datosCierres, tema = 'dark' }: PanelMACDProps) => {
  const datos = useMemo(() => {
    const resultado = calcularMacd(datosCierres.map((dato) => dato.cierre));
    return datosCierres.map((dato, indice) => ({ timestamp: dato.timestamp, macd: resultado.macd[indice], senal: resultado.senal[indice], histograma: resultado.histograma[indice] }));
  }, [datosCierres]);
  const colorTexto = tema === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="border-t border-gray-200 p-4">
      <h3 className="mb-2 text-xs font-medium" style={{ color: colorTexto }}>MACD (12, 26, 9)</h3>
      <div className="h-[150px] w-full">
        <ResponsiveContainer>
          <ComposedChart data={datos}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" hide />
            <YAxis tick={{ fill: colorTexto, fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="histograma" name="Histograma" fill="#10B981" isAnimationActive={false} />
            <Line type="monotone" dataKey="macd" name="MACD" stroke="#6366F1" dot={false} connectNulls isAnimationActive={false} />
            <Line type="monotone" dataKey="senal" name="Señal" stroke="#EF4444" dot={false} connectNulls isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default PanelMACD;
