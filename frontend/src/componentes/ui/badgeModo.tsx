export type BadgeModoProps = {
  modo: 'paper' | 'testnet' | 'live';
  conectado?: boolean;
  latencia?: number;
  ciclo?: number;
  nav?: number;
  onEmergencia?: () => void;
};

export const BadgeModo: React.FC<BadgeModoProps> = ({
  modo,
  conectado,
  latencia,
  ciclo,
  nav,
  onEmergencia,
}) => {
  const colores: Record<'paper' | 'testnet' | 'live', { color: string; bg: string; texto: string }> = {
    paper: { color: 'green', bg: '#d1e7dd', texto: 'PAPER' },
    testnet: { color: 'amber', bg: '#fff3cd', texto: 'TESTNET' },
    live: { color: 'red', bg: '#f8d7da', texto: 'LIVE' },
  };

  const config = colores[modo];
  const [mostrarBanner, setMostrarBanner] = React.useState(false);
  const [tooltip, setTooltip] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (modo === 'live') {
      setMostrarBanner(true);
    }
  }, [modo]);

  const handleEmergencia = () => {
    if (onEmergencia) {
      onEmergencia();
    }
  };

  return (
    <div className="relative w-24">
      {/* Badge del modo */}
      <button
        className="w-full px-3 py-2 rounded-md text-sm font-medium uppercase text-white focus:outline-none focus:ring-2 focus:ring-red-300"
        style={{ backgroundColor: config.bg, color: config.color }}
        title={config.texto}
        onMouseEnter={() => setTooltip(config.texto)}
        onMouseLeave={() => setTooltip(null)}
      >
        {config.texto}
      </button>

      {/* Banner fijo en LIVE */}
      {modo === 'live' && (
        <div
          className="fixed top-0 left-0 right-0 z-50 bg-red-600 text-white py-2 px-4 shadow-xl text-center invisible opacity-0 transition-opacity duration-300"
          aria-live="polite"
          style={{ visibility: mostrarBanner ? 'visible' : 'hidden' }>
            CONEXIÓN EN VIVO • Ciclo: {ciclo ?? 0} • NAV: {nav ?? 0} • Latencia: {latencia ? latencia + 'ms' : 'Desconocida'}
          </div>
        )
      }

      {/* Estado de conexión */}
      {conectado !== undefined && (
        <span className="mt-2 text-xs text-gray-500 connectado-badge">
          {conectado ? 'Conectado' : 'Desconectado'}
          {latencia !== undefined && ` • ${latencia}ms`}
        </span>
      )}

      {/* Botón emergencia */}
      {onEmergencia && (
        <button
          className="mt-2 w-full px-3 py-1 rounded-md text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300"
          onClick={handleEmergencia}
          aria-label="Parada de emergencia">
          CONFIRMAR
        </button>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-40 rounded-lg bg-gray-800 text-sm text-white p-3 shadow-lg z-50">
          <p className="font-medium mb-1">{tooltip}</p>
          <p className="text-xs opacity-80">Modo actual</p>
        </div>
      )}
    </div>
  );
};