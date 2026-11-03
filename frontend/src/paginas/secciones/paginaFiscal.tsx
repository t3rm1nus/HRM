import React from 'react';

const LOTES = [
  { id: 'L1', activo: 'BTC', cantidad: 0.0012, precio: 65000, fecha: '2026-02-10' },
];
const GANANCIAS = [
  { id: 'G1', activo: 'BTC', cantidad: 0.001, precioVenta: 66000, precioCompra: 65000, tenenciaDias: 30, ganancia: 10 },
];

export const PaginaFiscal: React.FC = () => {
  const [ejercicio, setEjercicio] = React.useState('2026');
  const exportar = () => {
    const csv = ['id,activo,ganancia', ...GANANCIAS.map((g) => `${g.id},${g.activo},${g.ganancia}`)].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'fiscal.csv';
    link.click();
    URL.revokeObjectURL(url);
  };
  const total = GANANCIAS.reduce((a, g) => a + g.ganancia, 0);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Fiscal</h1>
      <div className="mt-4 bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-800">
        Herramienta auxiliar, no asesoría fiscal.
      </div>
      <div className="mt-4">
        <label className="text-sm mr-2">Ejercicio</label>
        <select value={ejercicio} onChange={(e) => setEjercicio(e.target.value)} className="px-3 py-2 border rounded-lg text-sm">
          <option value="2026">2026</option>
          <option value="2025">2025</option>
        </select>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Lotes FIFO abiertos</h2>
        <ul className="text-sm">
          {LOTES.map((l) => (
            <li key={l.id}>{l.activo} {l.cantidad} @ {l.precio} ({l.fecha})</li>
          ))}
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Ganancias realizadas</h2>
        <table className="min-w-full text-sm">
          <thead>
            <tr><th className="text-left">Activo</th><th className="text-left">Cantidad</th><th className="text-left">Tenencia</th><th className="text-left">Ganancia</th></tr>
          </thead>
          <tbody>
            {GANANCIAS.map((g) => (
              <tr key={g.id}>
                <td>{g.activo}</td>
                <td>{g.cantidad}</td>
                <td>{g.tenenciaDias} días</td>
                <td>{g.ganancia}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Resumen anual</h2>
        <p className="text-sm">Total ganancias: {total}</p>
      </div>
      <div className="mt-6">
        <button type="button" onClick={exportar} className="px-4 py-2 bg-blue-600 text-white rounded-lg">
          Exportar CSV
        </button>
      </div>
    </div>
  );
};
