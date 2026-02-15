import os
import logging
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel

# Config logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_pickle(path):
    if os.path.exists(path):
        logging.info(f"Cargando {path}")
        return joblib.load(path)
    else:
        logging.warning(f"{path} no encontrado")
        return None

def load_huggingface_model(model_dir):
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        logging.info(f"Modelo HuggingFace cargado desde {model_dir}")
        return tokenizer, model
    logging.warning(f"{model_dir} no encontrado")
    return None, None

def mock_prices(tickers, start="2020-01-01", end="2025-01-01"):
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)
    for t in tickers:
        df[t] = np.cumsum(np.random.randn(len(dates)) * 2 + 0.1) + 100
    return df
