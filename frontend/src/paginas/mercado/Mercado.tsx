import React from 'react';

export const Mercado: React.FC = () => (
  <section className="mx-auto max-w-7xl space-y-4 p-4">
    <header>
      <h1 className="text-2xl font-bold text-slate-900">Mercado</h1>
      <p className="mt-1 text-sm text-slate-600">
        Datos de mercado proporcionados por el backend de HRM.
      </p>
    </header>
    <div className="rounded-lg border border-amber-300 bg-amber-50 p-5 text-amber-950">
      <h2 className="font-semibold">Fuente de mercado no conectada</h2>
      <p className="mt-2 text-sm">
        No se muestran velas ni precios hasta que el backend publique datos reales
        con su símbolo, mercado y marca temporal. Se han retirado los precios
        aleatorios que podían confundirse con cotizaciones reales.
      </p>
    </div>
  </section>
);

export default Mercado;
