import React, { useState, useCallback } from 'react';
import { PALABRAS_VERIFICACION } from '../dominio/modo';
import { BOTON_CONFIRMAR, BOTON_CANCELAR, BOTON_EMERGENCIA, BOTON_CERRAR } from '../constantes/boton';

interface ModalConfirmacionFuerteProps {
  titulo: string;
  textoPrimario: string;
  palabraEsperada: string;
  onConfirmar: () => void;
  onCancelar: () => void;
  activo: boolean;
  verificando: boolean;
}

export const ModalConfirmacionFuerte: React.FC<ModalConfirmacionFuerteProps> = ({
  titulo,
  textoPrimario,
  palabraEsperada,
  onConfirmar,
  onCancelar,
  activo,
  verificando,
}) => {
  const [escrito, setEscrito] = useState('');
  const [error, setError] = useState('');

  const verificar = useCallback(() => {
    if (escrito.toUpperCase() === palabraEsperada) {
      setError('');
      onConfirmar();
    } else {
      setError('Palabra incorrecta. Intenta de nuevo.');
    }
  }, [escrito, palabraEsperada, onConfirmar]);

  if (!activo) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-bold text-red-700">{titulo}</h3>
          <button
            type="button"
            className="rounded-lg p-1 hover:bg-gray-100"
            onClick={onCancelar}
            disabled={verificando}
          >
            {BOTON_CERRAR}
          </button>
        </div>

        <p className="mb-4 text-sm text-gray-600">{textoPrimario}</p>

        <div className="mb-4">
          <label
            htmlFor="palabra"
            className="mb-1 block text-xs font-medium text-gray-500"
          >
            Escribe la palabra de seguridad
          </label>
          <input
            id="palabra"
            type="text"
            autoComplete="off"
            value={escrito}
            onChange={(e) => {
              setEscrito(e.target.value);
              setError('');
            }}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-red-500 focus:outline-none focus:ring-2 focus:ring-red-200"
            placeholder="Escribe la palabra exacta..."
            disabled={verificando}
          />
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>

        <div className="flex gap-3">
          <button
            type="button"
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
            onClick={onCancelar}
            disabled={verificando}
          >
            {BOTON_CANCELAR}
          </button>
          <button
            type="button"
            className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
            onClick={verificar}
            disabled={verificando || escrito.toUpperCase() !== palabraEsperada}
          >
            {verificando ? 'Verificando...' : BOTON_CONFIRMAR}
          </button>
        </div>
      </div>
    </div>
  );
};
