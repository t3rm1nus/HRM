import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface LineaTemporalRegimenesProps {
  titulo: string;
  datos: { timestamp: string; regimen: 'PAPEL' | 'TESTNET' | 'LIVE'; activo: boolean }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  mostrarBandas?: boolean;
}

export const LineaTemporalRegimenes = ({ titulo, datos, height = 400, theme = 'dark', mostrarBandas = true }: LineaTemporalRegimenesProps) => {
  const valores = { PAPEL: 1, TESTNET: 2, LIVE: 3 };
  const datosGrafica = datos.map((dato) => ({ ...dato, nivel: dato.activo ? valores[dato.regimen] : 0 }));
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <AreaChart data={datosGrafica}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tick={{ fill: colorTexto, fontSize: 11 }} />
            <YAxis ticks={[0, 1, 2, 3]} tickFormatter={(valor: number) => ['Inactivo', 'PAPER', 'TESTNET', 'LIVE'][valor] ?? ''} tick={{ fill: colorTexto, fontSize: 11 }} />
            <Tooltip />
            <Area type="stepAfter" dataKey="nivel" name="Régimen" stroke={mostrarBandas ? '#10B981' : '#64748B'} fill={mostrarBandas ? '#10B981' : '#64748B'} fillOpacity={0.2} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default LineaTemporalRegimenes;
