import React from 'react';
import { PALABRAS_VERIFICACION } from '../dominio/modo';
import { useCambioModoSesion } from '../hooks/useCambioModoSesion';
import { ModalConfirmacionFuerte } from './modaConfirmacionFuerte';
import { SwitchModoSesion } from './ui/SwitchModoSesion';

const TEXTO_CONFIRMACION =
  'En el próximo arranque se borrarán la cartera y el historial de operaciones. Escribe CONFIRMAR para continuar.';

export const ControlModoSesion: React.FC = () => {
  const {
    modoConfigurado,
    reinicioPendiente,
    cargando,
    confirmacionPendiente,
    solicitarCambio,
    confirmarCambio,
    cancelarCambio,
  } = useCambioModoSesion();

  return (
    <>
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-500">Sesión</span>
        <SwitchModoSesion
          modoActual={modoConfigurado}
          reinicioPendiente={reinicioPendiente}
          onCambiarModo={solicitarCambio}
          deshabilitado={cargando}
        />
      </div>

      <ModalConfirmacionFuerte
        activo={confirmacionPendiente}
        verificando={cargando}
        titulo="Cambiar a sesión limpia"
        textoPrimario={TEXTO_CONFIRMACION}
        palabraEsperada={PALABRAS_VERIFICACION.CONFIRMAR}
        onConfirmar={confirmarCambio}
        onCancelar={cancelarCambio}
      />
    </>
  );
};
