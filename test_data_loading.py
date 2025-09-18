import asyncio
import sys
sys.path.append('.')
from l1_operational.binance_client import BinanceClient
from core.logging import logger

async def test_data():
    try:
        client = BinanceClient()
        data = await client.get_klines('BTCUSDT', '1m', 10)
        print(f'Data received: {len(data)} rows')
        if data:
            print(f'First row: {data[0]}')
        else:
            print('No data received')
        await client.close()
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_data())
