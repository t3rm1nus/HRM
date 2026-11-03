import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface HistogramaProps {
  titulo: string;
  rangos: { rango: string; count: number; monto?: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

export const Histograma = ({ titulo, rangos, height = 400, theme = 'dark' }: HistogramaProps) => {
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <BarChart data={rangos}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rango" tick={{ fill: colorTexto, fontSize: 11 }} />
            <YAxis tick={{ fill: colorTexto, fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" name="Frecuencia" fill="#F59E0B" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default Histograma;
