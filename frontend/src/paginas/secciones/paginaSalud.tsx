import React from 'react';

const COMPONENTES = [
  { nombre: 'L1', estado: 'ok' },
  { nombre: 'L2', estado: 'ok' },
  { nombre: 'L3', estado: 'degradado' },
];

export const PaginaSalud: React.FC = () => {
  const [logs, setLogs] = React.useState<string[]>([]);
  const [pausa, setPausa] = React.useState(false);
  const [nivel, setNivel] = React.useState('');
  const [modulo, setModulo] = React.useState('');
  const [busqueda, setBusqueda] = React.useState('');

  React.useEffect(() => {
    const interval = setInterval(() => {
      if (pausa) return;
      setLogs((prev) => {
        const next = [...prev, `[INFO] ${new Date().toISOString()} evento`];
        return next.slice(-10000);
      });
    }, 200);
    return () => clearInterval(interval);
  }, [pausa]);

  const filtrados = logs.filter((l) => {
    if (nivel && !l.includes(nivel)) return false;
    if (modulo && !l.includes(modulo)) return false;
    if (busqueda && !l.toLowerCase().includes(busqueda.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Salud del sistema</h1>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {COMPONENTES.map((c) => (
          <div key={c.nombre} className="bg-white border rounded-xl p-4">
            <p className="font-semibold">{c.nombre}</p>
            <p className={`text-sm ${c.estado === 'ok' ? 'text-green-600' : 'text-yellow-600'}`}>{c.estado}</p>
          </div>
        ))}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Frescura de datos de mercado</h2>
        <p className="text-sm">Última actualización: hace 2s</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Conexión al exchange</h2>
        <p className="text-sm text-green-600">Conectado</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Errores y recuperaciones</h2>
        <p className="text-sm">Errores: 3 | Recuperaciones: 3</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Actividad MessageBus</h2>
        <p className="text-sm">3 temas activos</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <div className="flex items-center gap-2 mb-2">
          <h2 className="font-semibold">Logs en vivo</h2>
          <button type="button" onClick={() => setPausa((p) => !p)} className="px-2 py-1 text-xs bg-gray-100 rounded">
            {pausa ? 'Reanudar' : 'Pausar'}
          </button>
          <input placeholder="nivel" value={nivel} onChange={(e) => setNivel(e.target.value)} className="px-2 py-1 border rounded text-xs" />
          <input placeholder="modulo" value={modulo} onChange={(e) => setModulo(e.target.value)} className="px-2 py-1 border rounded text-xs" />
          <input placeholder="buscar" value={busqueda} onChange={(e) => setBusqueda(e.target.value)} className="px-2 py-1 border rounded text-xs" />
        </div>
        <div className="h-64 overflow-auto bg-black text-green-200 text-xs font-mono p-2 rounded">
          {filtrados.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </div>
    </div>
  );
};
