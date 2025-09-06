import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

steps = [
    "l3_strategy/data_fetcher.py",
    "l3_strategy/l3_processor.py",
    "l3_strategy/l2_processor.py",
    "l3_strategy/l1_processor.py",
    "l3_strategy/hrm_bl.py"
]

for s in steps:
    logging.info(f"Ejecutando {s}")
    os.system(f"python {s}")
