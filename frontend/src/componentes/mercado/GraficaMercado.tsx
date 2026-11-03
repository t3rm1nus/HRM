import { useMemo } from 'react';
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { calcularMediaMovil } from '../../dominio/indicadores/calculos';

export interface GraficaMercadoProps {
  simbolo: string;
  velas: { timestamp: string; open: number; high: number; low: number; close: number; volume: number }[];
  timeframe: string;
  capas?: { ma50?: boolean; ma200?: boolean; bollinger?: boolean; soportes?: boolean; resistencias?: boolean };
  tema?: 'light' | 'dark';
}

export const GraficaMercado = ({ simbolo, velas, timeframe, capas = {}, tema = 'dark' }: GraficaMercadoProps) => {
  const datos = useMemo(() => {
    const cierres = velas.map((vela) => vela.close);
    const ma50 = calcularMediaMovil(cierres, 50);
    const ma200 = calcularMediaMovil(cierres, 200);
    return velas.map((vela, indice) => ({ ...vela, ma50: ma50[indice], ma200: ma200[indice] }));
  }, [velas]);
  const colorTexto = tema === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border p-4">
      <header className="mb-2 flex items-center justify-between"><h3 className="text-sm font-medium" style={{ color: colorTexto }}>{simbolo}</h3><span className="text-xs text-gray-500">{timeframe}</span></header>
      <div className="h-[400px] w-full">
        <ResponsiveContainer>
          <ComposedChart data={datos}>
            <CartesianGrid strokeDasharray="3 3" stroke={tema === 'dark' ? '#374151' : '#E5E7EB'} />
            <XAxis dataKey="timestamp" tick={{ fill: colorTexto, fontSize: 11 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fill: colorTexto, fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="close" name="Cierre" stroke="#10B981" dot={false} isAnimationActive={false} />
            {capas.ma50 && <Line type="monotone" dataKey="ma50" name="MA 50" stroke="#F59E0B" dot={false} connectNulls isAnimationActive={false} />}
            {capas.ma200 && <Line type="monotone" dataKey="ma200" name="MA 200" stroke="#3B82F6" dot={false} connectNulls isAnimationActive={false} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {(capas.bollinger || capas.soportes || capas.resistencias) && <p className="mt-2 text-xs text-amber-600">Las bandas de Bollinger y los niveles de soporte/resistencia aún no se calculan: no se muestran para evitar indicadores ficticios.</p>}
    </section>
  );
};

export default GraficaMercado;
