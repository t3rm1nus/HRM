import pandas as pd
from utils import logging

l3 = pd.read_json("data/l3_output.json")
l2 = pd.read_csv("data/l2_signals.csv")
l1 = pd.read_csv("data/l1_scores.csv")

# Mock de integración jerárquica: pesos finales
portfolio = pd.DataFrame({
    "ticker": l3.iloc[0]["asset_allocation"].keys(),
    "weight_final": [0.4,0.2,0.2,0.1,0.1]  # Mock de cálculo BL jerárquico
})
portfolio.to_csv("models/L3/portfolio/final_portfolio.csv", index=False)
logging.info("Portfolio final guardado en models/L3/portfolio/final_portfolio.csv")
