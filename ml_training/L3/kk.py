import pandas as pd

path = "data/datos_para_modelos_l3/portfolio/BTC-USD.csv"
df = pd.read_csv(path)
print(df.columns)