export interface MapaCalorProps {
  titulo: string;
  datos: { x: string; y: string; valor: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

function obtenerColor(valor: number, minimo: number, maximo: number): string {
  const intensidad = maximo === minimo ? 0.5 : (valor - minimo) / (maximo - minimo);
  const rojo = Math.round(239 * intensidad + 16 * (1 - intensidad));
  const verde = Math.round(68 * intensidad + 185 * (1 - intensidad));
  const azul = Math.round(68 * intensidad + 129 * (1 - intensidad));
  return `rgb(${rojo}, ${verde}, ${azul})`;
}

export const MapaCalor = ({ titulo, datos, height = 400, theme = 'dark' }: MapaCalorProps) => {
  const ejesX = [...new Set(datos.map((dato) => dato.x))];
  const ejesY = [...new Set(datos.map((dato) => dato.y))];
  const valores = datos.map((dato) => dato.valor).filter(Number.isFinite);
  const minimo = Math.min(...valores);
  const maximo = Math.max(...valores);
  const celdas = new Map(datos.map((dato) => [`${dato.x}\u0000${dato.y}`, dato]));
  const colorTexto = theme === 'dark' ? '#F9FAFB' : '#111827';
  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-3 text-sm font-medium" style={{ color: colorTexto }}>{titulo}</h3>
      <div className="overflow-auto" style={{ height }}>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${Math.max(ejesX.length, 1)}, minmax(2.5rem, 1fr))` }}>
          {ejesY.flatMap((ejeY) => ejesX.map((ejeX) => {
            const celda = celdas.get(`${ejeX}\u0000${ejeY}`);
            return <div key={`${ejeX}-${ejeY}`} className="flex min-h-10 items-center justify-center rounded text-xs text-white" title={celda ? `${ejeX} / ${ejeY}: ${celda.valor}` : `${ejeX} / ${ejeY}`} style={{ backgroundColor: celda ? obtenerColor(celda.valor, minimo, maximo) : '#64748B33' }}>{celda?.valor ?? '—'}</div>;
          }))}
        </div>
      </div>
    </section>
  );
};

export default MapaCalor;
