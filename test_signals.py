import asyncio
import sys
sys.path.append('.')
from l2_tactic.signal_generator import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.models import TacticalSignal
import pandas as pd
import numpy as np

async def test_signals():
    try:
        # Create test state
        test_state = {
            'market_data': {
                'BTCUSDT': pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [105, 106, 107],
                    'low': [95, 96, 97],
                    'close': [103, 104, 105],
                    'volume': [1000, 1100, 1200]
                }),
                'ETHUSDT': pd.DataFrame({
                    'open': [200, 201, 202],
                    'high': [210, 211, 212],
                    'low': [190, 191, 192],
                    'close': [205, 206, 207],
                    'volume': [2000, 2100, 2200]
                })
            }
        }

        # Create L2 processor
        config = L2Config()
        processor = L2TacticProcessor(config)

        # Process signals
        signals = await processor.process_signals(test_state)
        print(f'Raw signals returned: {len(signals)}')
        for i, s in enumerate(signals):
            print(f'  Signal {i}: type={type(s)}, value={s}')

        # Filter valid signals
        valid_signals = [s for s in signals if isinstance(s, TacticalSignal)]
        print(f'Valid signals: {len(valid_signals)}')
        for i, s in enumerate(valid_signals):
            print(f'  Valid signal {i}: {s}')

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_signals())
