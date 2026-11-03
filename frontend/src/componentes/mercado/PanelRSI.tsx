import { useMemo } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { calcularRsi } from '../../dominio/indicadores/calculos';

export interface PanelRSIProps {
  datosCierres: { timestamp: string; cierre: number }[];
  timeframe: string;
  tema?: 'light' | 'dark';
}

export const PanelRSI = ({ datosCierres, tema = 'dark' }: PanelRSIProps) => {
  const datos = useMemo(() => {
    const rsi = calcularRsi(datosCierres.map((dato) => dato.cierre));
    return datosCierres.map((dato, indice) => ({ timestamp: dato.timestamp, rsi: rsi[indice] }));
  }, [datosCierres]);
  const colorTexto = tema === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="border-t border-gray-200 p-4">
      <h3 className="mb-2 text-xs font-medium" style={{ color: colorTexto }}>RSI (14)</h3>
      <div className="h-[150px] w-full">
        <ResponsiveContainer>
          <LineChart data={datos}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" hide />
            <YAxis domain={[0, 100]} ticks={[0, 30, 50, 70, 100]} tick={{ fill: colorTexto, fontSize: 10 }} />
            <Tooltip />
            <ReferenceLine y={70} stroke="#EF4444" strokeDasharray="4 4" />
            <ReferenceLine y={30} stroke="#10B981" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="rsi" name="RSI" stroke="#F59E0B" dot={false} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-gray-500">Sobrecompra: 70 | Sobreventa: 30</p>
    </section>
  );
};

export default PanelRSI;
