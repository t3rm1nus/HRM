import React from 'react';

const INFORMES = [
  { id: 'B1', fecha: '2026-02-10', simbodo: 'BTCUSDT', perfil: 'conservador', retorno: 0.12, drawdown: -0.05 },
  { id: 'B2', fecha: '2026-02-11', simbodo: 'ETHUSDT', perfil: 'agresivo', retorno: 0.08, drawdown: -0.08 },
];

export const PaginaBacktesting: React.FC = () => {
  const [form, setForm] = React.useState({ inicio: '', fin: '', simbodo: '', perfil: 'conservador' });
  const [progreso, setProgreso] = React.useState(0);
  const [ejecutando, setEjecutando] = React.useState(false);
  const [seleccion, setSeleccion] = React.useState<string[]>([]);
  const lanzar = () => {
    if (!form.inicio || !form.fin || !form.simbodo) return;
    setEjecutando(true);
    setProgreso(0);
    const timer = setInterval(() => {
      setProgreso((p) => {
        if (p >= 100) {
          clearInterval(timer);
          setEjecutando(false);
          return 100;
        }
        return p + 20;
      });
    }, 300);
  };
  const comparar = seleccion.length === 2
    ? INFORMES.find((i) => i.id === seleccion[0]) && INFORMES.find((i) => i.id === seleccion[1])
    : null;
  const a = INFORMES.find((i) => i.id === seleccion[0]);
  const b = INFORMES.find((i) => i.id === seleccion[1]);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Backtesting</h1>
      <div className="mt-4 bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-800">
        Aviso: posibles sesgos de look-ahead, fuga de datos y costes no modelados.
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Lanzar backtest</h2>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
          <input type="date" value={form.inicio} onChange={(e) => setForm({ ...form, inicio: e.target.value })} className="px-3 py-2 border rounded-lg text-sm" />
          <input type="date" value={form.fin} onChange={(e) => setForm({ ...form, fin: e.target.value })} className="px-3 py-2 border rounded-lg text-sm" />
          <input placeholder="Símbodo" value={form.simbodo} onChange={(e) => setForm({ ...form, simbodo: e.target.value })} className="px-3 py-2 border rounded-lg text-sm" />
          <select value={form.perfil} onChange={(e) => setForm({ ...form, perfil: e.target.value })} className="px-3 py-2 border rounded-lg text-sm">
            <option value="conservador">conservador</option>
            <option value="agresivo">agresivo</option>
          </select>
        </div>
        <button type="button" onClick={lanzar} disabled={ejecutando} className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50">
          Lanzar
        </button>
        {ejecutando && (
          <div className="mt-3">
            <div className="h-2 bg-gray-200 rounded">
              <div className="h-2 bg-blue-500 rounded" style={{ width: `${progreso}%` }} />
            </div>
            <p className="text-xs mt-1">{progreso}%</p>
          </div>
        )}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Informes previos</h2>
        <ul className="text-sm space-y-1">
          {INFORMES.map((i) => (
            <li key={i.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={seleccion.includes(i.id)}
                onChange={(e) => {
                  if (e.target.checked) setSeleccion((s) => [...s, i.id].slice(-2));
                  else setSeleccion((s) => s.filter((x) => x !== i.id));
                }}
              />
              {i.id} - {i.simbodo} - {i.perfil} (ret {i.retorno}, dd {i.drawdown})
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Visor (equity / drawdown / distribución)</h2>
        <p className="text-sm text-gray-600">Gráficos simulados</p>
      </div>
      {a && b && (
        <div className="mt-6 bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Comparación</h2>
          <table className="min-w-full text-sm">
            <thead>
              <tr><th className="text-left">Métrica</th><th className="text-left">{a.id}</th><th className="text-left">{b.id}</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Retorno</td>
                <td className={a.retorno !== b.retorno ? 'text-blue-600 font-semibold' : ''}>{a.retorno}</td>
                <td className={a.retorno !== b.retorno ? 'text-blue-600 font-semibold' : ''}>{b.retorno}</td>
              </tr>
              <tr>
                <td>Drawdown</td>
                <td className={a.drawdown !== b.drawdown ? 'text-blue-600 font-semibold' : ''}>{a.drawdown}</td>
                <td className={a.drawdown !== b.drawdown ? 'text-blue-600 font-semibold' : ''}>{b.drawdown}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
