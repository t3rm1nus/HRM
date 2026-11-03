import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface GraficaLineaProps {
  titulo: string;
  datos: { timestamp: string; valor: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  showGrid?: boolean;
  crosshairSource?: string;
  onPointClick?: (valor: number, timestamp: string) => void;
}

export const GraficaLinea = ({ titulo, datos, height = 400, theme = 'dark', showGrid = true, onPointClick }: GraficaLineaProps) => {
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={datos} onClick={(evento) => {
            const indice = Number(evento?.activeTooltipIndex);
            const punto = Number.isInteger(indice) ? datos[indice] : undefined;
            if (punto) onPointClick?.(punto.valor, punto.timestamp);
          }}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#E5E7EB'} />}
            <XAxis dataKey="timestamp" tick={{ fill: colorTexto, fontSize: 11 }} />
            <YAxis tick={{ fill: colorTexto, fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="valor" name={titulo} stroke="#2563EB" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default GraficaLinea;
