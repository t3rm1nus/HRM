import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface GraficaBarrasProps {
  titulo: string;
  datos: { timestamp: string; valor: number; categoria?: string }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  showGrid?: boolean;
  crosshairSource?: string;
  onBarClick?: (valor: number, categoria: string, timestamp: string) => void;
}

export const GraficaBarras = ({ titulo, datos, height = 400, theme = 'dark', showGrid = true, onBarClick }: GraficaBarrasProps) => {
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  const datosGrafica = datos.map((dato) => ({ ...dato, ejeX: dato.categoria ?? dato.timestamp }));
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <BarChart data={datosGrafica} onClick={(evento) => { const punto = evento?.activePayload?.[0]?.payload as (typeof datos)[number] | undefined; if (punto) onBarClick?.(punto.valor, punto.categoria ?? '', punto.timestamp); }}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#E5E7EB'} />}
            <XAxis dataKey="ejeX" tick={{ fill: colorTexto, fontSize: 11 }} />
            <YAxis tick={{ fill: colorTexto, fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="valor" name={titulo} fill="#3B82F6" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default GraficaBarras;
