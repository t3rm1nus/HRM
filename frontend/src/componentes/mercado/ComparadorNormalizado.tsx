import { useMemo } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface ComparadorNormalizadoProps {
  btcDatos: { timestamp: string; precio: number }[];
  ethDatos: { timestamp: string; precio: number }[];
  width?: number;
  height?: number;
  tema?: 'light' | 'dark';
}

export const ComparadorNormalizado = ({ btcDatos, ethDatos, height = 200, tema = 'dark' }: ComparadorNormalizadoProps) => {
  const datos = useMemo(() => {
    const normalizar = (valores: typeof btcDatos) => {
      const base = valores.find((dato) => dato.precio > 0)?.precio;
      return new Map(valores.map((dato) => [dato.timestamp, base ? (dato.precio / base) * 100 : null]));
    };
    const btc = normalizar(btcDatos);
    const eth = normalizar(ethDatos);
    return [...new Set([...btc.keys(), ...eth.keys()])].sort().map((timestamp) => ({ timestamp, btc: btc.get(timestamp) ?? null, eth: eth.get(timestamp) ?? null }));
  }, [btcDatos, ethDatos]);
  const colorTexto = tema === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>Comparación BTC/ETH (base 100)</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={datos}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tick={{ fill: colorTexto, fontSize: 10 }} />
            <YAxis tick={{ fill: colorTexto, fontSize: 10 }} />
            <Tooltip />
            <Line dataKey="btc" name="BTC" stroke="#F59E0B" dot={false} connectNulls isAnimationActive={false} />
            <Line dataKey="eth" name="ETH" stroke="#3B82F6" dot={false} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default ComparadorNormalizado;
