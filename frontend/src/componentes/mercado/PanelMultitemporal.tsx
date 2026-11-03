import React from 'react';

export interface PanelMultitemporalProps {
  simbolo: string;
  timeframes: string[];
  setTimeframe: (timeframe: string) => void;
}

export const PanelMultitemporal: React.FC<PanelMultitemporalProps> = ({
  simbolo,
  timeframes,
  setTimeframe,
}) => {
  return (
    <div className="space-y-2">
      {timeframes.map((tf) => (
        <button
          key={tf}
          onClick={() => setTimeframe(tf)}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
          aria-pressed={tf === '1m'} /* Simplified */
        >
          {tf}
        </button>
      ))}
    </div>
  );
};

export default PanelMultitemporal;