import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';

export interface DonutProps {
  titulo: string;
  datos: { nombre: string; valor: number; color?: string }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

export const Donut = ({ titulo, datos, height = 300, theme = 'dark' }: DonutProps) => {
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie data={datos} dataKey="valor" nameKey="nombre" innerRadius="55%" outerRadius="82%" paddingAngle={2}>
              {datos.map((dato, indice) => <Cell key={`${dato.nombre}-${indice}`} fill={dato.color ?? ['#10B981', '#3B82F6', '#F59E0B', '#EF4444'][indice % 4]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default Donut;
