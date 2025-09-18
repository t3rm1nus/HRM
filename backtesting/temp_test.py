import asyncio
import sys
sys.path.append('..')
from backtesting.main import HRMBacktester

async def test_backtest():
    backtester = HRMBacktester()
    print('Backtester initialized successfully')
    print('Config:', backtester.config.keys())
    print('Data collector:', type(backtester.data_collector).__name__)
    print('Strategy tester:', type(backtester.strategy_tester).__name__)
    print('Performance analyzer:', type(backtester.performance_analyzer).__name__)
    print('Report generator:', type(backtester.report_generator).__name__)
    print('Backtesting system components are properly initialized')

asyncio.run(test_backtest())
