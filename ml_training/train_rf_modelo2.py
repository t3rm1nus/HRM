import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Paths
data_dir = Path(__file__).parent.parent / "data"
train_path = data_dir / "btc_features_train.csv"
test_path = data_dir / "btc_features_test.csv"
model_path = Path(__file__).parent / "rf_modelo2.pkl"

# Cargar datos
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Características (excluir timestamp y columna objetivo)
FEATURE_COLS = [c for c in train_df.columns if c not in ["timestamp", "close_5m", "delta_close_5m"]]

X_train = train_df[FEATURE_COLS]
y_train = train_df["close_5m"]

X_test = test_df[FEATURE_COLS]
y_test = test_df["close_5m"]

# Entrenar modelo
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluación
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE Test: {mse:.4f}")

# Guardar modelo
joblib.dump(rf, model_path)
print(f"Modelo guardado en {model_path}")
