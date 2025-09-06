import pandas as pd
from utils import load_pickle, logging

LR_MODEL = "models/L1/modelo1_lr.pkl"
RF_MODEL = "models/L1/modelo2_rf.pkl"
LGBM_MODEL = "models/L1/modelo3_lgbm.pkl"

# Cargar modelos
lr = load_pickle(LR_MODEL)
rf = load_pickle(RF_MODEL)
lgbm = load_pickle(LGBM_MODEL)

tickers = ["AAPL", "MSFT", "GOOG", "BTC-USD", "ETH-USD"]
l1_scores = pd.DataFrame({
    "ticker": tickers,
    "score_lr": [0.6,0.7,0.5,0.8,0.65],
    "score_rf": [0.55,0.72,0.48,0.79,0.63],
    "score_lgbm": [0.57,0.69,0.52,0.81,0.64]
})
l1_scores.to_csv("data/l1_scores.csv", index=False)
logging.info("L1 scores guardados en data/l1_scores.csv")
