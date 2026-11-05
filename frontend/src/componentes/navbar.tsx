import React from 'react';
import { useModoStore } from '../estado/storeModo';
import { BadgeModo } from './ui/badgeModo';
import { ControlModoSesion } from './ControlModoSesion';
import { ModalConfirmacionFuerte } from './modaConfirmacionFuerte';
import { ModoAplicacion, PALABRAS_VERIFICACION } from '../dominio/modo';


export const Navbar: React.FC = () => {
  const {
    modo,
    estadoConexion,
    latenciaMs,
    deteniendo,
    alarmeActivado,
    cicloCount,
    modoConfirmado,
    confirmarDetencionEmergencia,
    cancelarDetencionEmergencia,
  } = useModoStore();

  const [confirmacionActiva, setConfirmacionActiva] = React.useState(false);
  const [detenerConfirmado, setDetenerConfirmado] = React.useState(false);

  const handleEmergencia = () => {
    if (deteniendo) {
      setDetenerConfirmado(true);
      return;
    }
    setConfirmacionActiva(true);
  };

  const confirmarEmergencia = () => {
    setConfirmacionActiva(false);
    setDetenerConfirmado(false);
    confirmarDetencionEmergencia();
  };

  const cancelarEmergencia = () => {
    setConfirmacionActiva(false);
    setDetenerConfirmado(false);
    cancelarDetencionEmergencia();
  };

  const estadoConexionTexto = estadoConexion === 'conectado' ? 'Conectado' : estadoConexion === 'desconectado' ? 'Desconectado' : 'Pendiente';

  return (
    <>
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            {modoConfirmado ? (
              <BadgeModo modo={modo === ModoAplicacion.PAPERA ? 'paper' : modo} />
            ) : (
              <span
                className="px-3 py-2 rounded-lg text-sm font-semibold bg-gray-200 text-gray-600"
                title="El backend no ha confirmado el modo de operación"
              >
                MODO SIN CONFIRMAR
              </span>
            )}
            <ControlModoSesion />
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span className={`flex items-center gap-1.5 ${estadoConexion === 'conectado' ? 'text-green-600' : estadoConexion === 'desconectado' ? 'text-red-600' : 'text-amber-600'}`}>
                <span className={`w-2 h-2 rounded-full ${estadoConexion === 'conectado' ? 'bg-green-500' : estadoConexion === 'desconectado' ? 'bg-red-500' : 'bg-amber-500'}`} />
                {estadoConexionTexto}
              </span>
              {latenciaMs >= 0 && (
                <span className="flex items-center gap-1.5">
                  ⏱️
                  {latenciaMs}ms
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50">
              📊
              <span className="text-sm font-semibold">Ciclo {cicloCount}</span>
            </div>

            <div
              className={`flex items-center justify-center w-3 h-3 rounded-full ${alarmeActivado ? 'bg-red-500' : 'bg-gray-300'}`}
              title={alarmeActivado ? 'Alarme activado' : 'Sin alarme'}
            />

            <button
              type="button"
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-gray-200 text-gray-500 cursor-not-allowed"
              onClick={handleEmergencia}
              disabled
              title="No disponible: el backend aún no expone un endpoint de parada"
            >
              🛑
              <span className="hidden sm:inline">{deteniendo ? 'Detenido' : 'Detener'}</span>
            </button>
          </div>
        </div>
      </header>

      <ModalConfirmacionFuerte
        activo={confirmacionActiva}
        verificando={false}
        titulo="Confirmar emergencia"
        textoPrimario="La operación se detendrá inmediatamente. Escribe CONFIRMAR para continuar."
        palabraEsperada={PALABRAS_VERIFICACION.CONFIRMAR}
        onConfirmar={confirmarEmergencia}
        onCancelar={cancelarEmergencia}
      />

      <ModalConfirmacionFuerte
        activo={detenerConfirmado}
        verificando={true}
        titulo="¿Seguro que deseas detener todo?"
        textoPrimario="Esta acción es irreversible. Escribe CONFIRMAR para confirmar la detención de emergencia."
        palabraEsperada={PALABRAS_VERIFICACION.CONFIRMAR}
        onConfirmar={confirmarEmergencia}
        onCancelar={cancelarEmergencia}
      />
    </>
  );
};
