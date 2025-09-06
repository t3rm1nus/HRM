import asyncio
from datetime import datetime, timedelta
import pandas as pd
from l2_tactic.models import L2State
from core.state_manager import log_cycle_data

async def run_test_with_l2state():
    state = {
        'l2': L2State(),
        'ordenes': [{'id':1,'status':'rejected'},{'id':2,'status':'filled'}],
        'portfolio': {'BTCUSDT':0.0,'ETHUSDT':0.0,'USDT':3000.0}
    }
    state['l2'].signals = [{'symbol':'BTCUSDT','action':'buy'}]
    start = pd.Timestamp.utcnow() - pd.Timedelta(seconds=1)
    await log_cycle_data(state, cycle_id=1, ciclo_start=start)
    print('L2State test done')

async def run_test_with_dict():
    state = {
        'l2': {'signals':[{'symbol':'ETHUSDT','action':'sell'}]},
        'ordenes': [{'id':1,'status':'rejected'}],
        'portfolio': {'BTCUSDT':0.0,'ETHUSDT':0.0,'USDT':3000.0}
    }
    start = pd.Timestamp.utcnow() - pd.Timedelta(seconds=2)
    await log_cycle_data(state, cycle_id=2, ciclo_start=start)
    print('dict l2 test done')

async def main():
    await run_test_with_l2state()
    await run_test_with_dict()

if __name__ == '__main__':
    asyncio.run(main())
