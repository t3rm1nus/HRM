import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import joblib

# Paths
data_dir = Path(__file__).parent.parent / "data"
train_path = data_dir / "btc_features_train.csv"
test_path = data_dir / "btc_features_test.csv"
model_path = Path(__file__).parent / "lgbm_modelo3.pkl"

# Cargar datos
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Crear variable objetivo: delta_close_5m > 0 -> 1, else 0
train_df['target'] = (train_df['delta_close_5m'] > 0).astype(int)
test_df['target'] = (test_df['delta_close_5m'] > 0).astype(int)

# Características
FEATURE_COLS = [c for c in train_df.columns if c not in ["timestamp", "delta_close_5m", "close_5m", "target"]]

X_train = train_df[FEATURE_COLS]
y_train = train_df["target"]

X_test = test_df[FEATURE_COLS]
y_test = test_df["target"]

# Crear dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parámetros
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
    'random_state': 42
}

# Entrenar con callback de early stopping
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# Predicción y evaluación
y_pred = gbm.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_labels)
print(f"Accuracy Test: {acc:.4f}")

# Guardar modelo
joblib.dump(gbm, model_path)
print(f"Modelo guardado en {model_path}")
