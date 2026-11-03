import React from 'react';

const MODELOS = [
  { nivel: 'L1', ruta: 'models/l1/model.pt', existe: false, tamano: 0, hash: '', ultimaCarga: '', estado: 'ausente', accion: 'Entrenar L1' },
  { nivel: 'L2', ruta: 'models/l2/model.pt', existe: false, tamano: 0, hash: '', ultimaCarga: '', estado: 'ausente', accion: 'Entrenar L2' },
  { nivel: 'L3', ruta: 'models/l3/model.pt', existe: false, tamano: 0, hash: '', ultimaCarga: '', estado: 'ausente', accion: 'Entrenar L3' },
];

export const PaginaModelos: React.FC = () => {
  const disponibles = MODELOS.filter((m) => m.existe).length;
  const copiar = (ruta: string) => navigator.clipboard?.writeText(ruta);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Modelos</h1>
      <p className="mt-2 text-sm text-gray-600">Disponibles: {disponibles} de {MODELOS.length}</p>
      <div className="mt-6 bg-white border rounded-xl overflow-hidden">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left">Nivel</th>
              <th className="px-4 py-3 text-left">Ruta esperada</th>
              <th className="px-4 py-3 text-left">Existe</th>
              <th className="px-4 py-3 text-left">Tamaño</th>
              <th className="px-4 py-3 text-left">Hash</th>
              <th className="px-4 py-3 text-left">Última carga</th>
              <th className="px-4 py-3 text-left">Estado</th>
              <th className="px-4 py-3 text-left">Acción</th>
            </tr>
          </thead>
          <tbody>
            {MODELOS.map((m) => (
              <tr key={m.nivel} className={m.existe ? '' : 'bg-red-50'}>
                <td className="px-4 py-3">{m.nivel}</td>
                <td className="px-4 py-3">
                  {m.ruta}
                  <button type="button" onClick={() => copiar(m.ruta)} className="ml-2 text-xs text-blue-600 hover:underline">
                    Copiar
                  </button>
                </td>
                <td className="px-4 py-3">{m.existe ? 'Sí' : 'No'}</td>
                <td className="px-4 py-3">{m.tamano}</td>
                <td className="px-4 py-3">{m.hash || '-'}</td>
                <td className="px-4 py-3">{m.ultimaCarga || '-'}</td>
                <td className="px-4 py-3">{m.estado}</td>
                <td className="px-4 py-3">{m.accion}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
