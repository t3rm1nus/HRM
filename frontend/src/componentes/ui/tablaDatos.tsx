import React from 'react';

export interface Columna<T> {
  clave: string;
  titulo: string;
  ancho?: string;
  sortable?: boolean;
  render?: (valor: any, fila: T) => React.ReactNode;
}

export interface TablaDatosProps<T> {
  datos: T[];
  columnas: Columna<T>[];
  porPagina?: number;
  exportable?: boolean;
  onExportar?: () => void;
  cargando?: boolean;
  vacio?: string;
  error?: string;
  ordenable?: boolean;
  filtrable?: boolean;
  filtro?: string;
  onFiltroChange?: (filtro: string) => void;
  onOrdenChange?: (columna: string, direccion: 'asc' | 'desc') => void;
}

export const TablaDatos = <T extends Record<string, any>>({
  datos,
  columnas,
  porPagina = 10,
  exportable = false,
  onExportar,
  cargando = false,
  vacio = 'No hay datos para mostrar',
  error,
  ordenable = false,
  filtrable = false,
  filtro = '',
  onFiltroChange,
  onOrdenChange,
}: TablaDatosProps<T>) => {
  const [paginaActual, setPaginaActual] = React.useState(1);
  const [ordenColumna, setOrdenColumna] = React.useState<string | null>(null);
  const [ordenDireccion, setOrdenDireccion] = React.useState<'asc' | 'desc'>('asc');

  const handleOrden = (columna: string) => {
    if (!ordenable) return;
    const nuevaDireccion = ordenColumna === columna && ordenDireccion === 'asc' ? 'desc' : 'asc';
    setOrdenColumna(columna);
    setOrdenDireccion(nuevaDireccion);
    onOrdenChange?.(columna, nuevaDireccion);
  };

  const handleExportarCSV = () => {
    if (onExportar) {
      onExportar();
      return;
    }
    const csv = [
      columnas.map((c) => c.titulo).join(','),
      ...datos.map((fila) =>
        columnas.map((c) => {
          const valor = fila[c.clave];
          return typeof valor === 'string' ? `"${valor}"` : valor;
        }).join(',')
      ),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'datos.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const datosPaginados = datos.slice(
    (paginaActual - 1) * porPagina,
    paginaActual * porPagina
  );
  const totalPaginas = Math.ceil(datos.length / porPagina);

  if (cargando) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-8">
        <p className="text-sm text-red-600">{error}</p>
      </div>
    );
  }

  if (datos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8">
        <p className="text-sm text-gray-500">{vacio}</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      {filtrable && (
        <div className="p-4 border-b border-gray-200">
          <input
            type="text"
            value={filtro}
            onChange={(e) => onFiltroChange?.(e.target.value)}
            placeholder="Filtrar..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-200"
            aria-label="Filtrar datos"
          />
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columnas.map((col) => (
                <th
                  key={col.clave}
                  className={`px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${ordenable && col.sortable ? 'cursor-pointer hover:bg-gray-100' : ''}`}
                  onClick={() => col.sortable && handleOrden(col.clave)}
                  style={col.ancho ? { width: col.ancho } : undefined}
                >
                  <div className="flex items-center gap-1">
                    {col.titulo}
                    {ordenable && col.sortable && ordenColumna === col.clave && (
                      <span>{ordenDireccion === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {datosPaginados.map((fila, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                {columnas.map((col) => (
                  <td key={col.clave} className="px-4 py-3 text-sm text-gray-600">
                    {col.render ? col.render(fila[col.clave], fila) : fila[col.clave]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
        <span className="text-sm text-gray-500">
          {datos.length} registros
        </span>
        <div className="flex items-center gap-2">
          {exportable && (
            <button
              type="button"
              className="px-3 py-1.5 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
              onClick={handleExportarCSV}
            >
              Exportar CSV
            </button>
          )}
          <div className="flex gap-1">
            <button
              type="button"
              className="px-3 py-1.5 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
              onClick={() => setPaginaActual((p) => Math.max(1, p - 1))}
              disabled={paginaActual === 1}
            >
              ←
            </button>
            <span className="px-3 py-1.5 text-sm text-gray-700">
              {paginaActual} / {totalPaginas}
            </span>
            <button
              type="button"
              className="px-3 py-1.5 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
              onClick={() => setPaginaActual((p) => Math.min(totalPaginas, p + 1))}
              disabled={paginaActual === totalPaginas}
            >
              →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TablaDatos;
