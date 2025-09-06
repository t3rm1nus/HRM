import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import optuna

# ======== 1. CARGAR DATOS ========
data_folder = 'data/datos_para_modelos_l3/modelo1_regime_detector'
data_file = os.path.join(data_folder, 'market_data_features_labels.csv')

data = pd.read_csv(data_file, parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# ======== 2. FEATURES DERIVADOS ========
if 'return' not in data.columns:
    data['return'] = data['close'].pct_change()
if 'log_return' not in data.columns:
    data['log_return'] = np.log1p(data['return'])

windows = [5, 15, 30, 60, 120]
for w in windows:
    data[f'volatility_{w}'] = data['return'].rolling(w, min_periods=1).std()
    data[f'return_{w}'] = data['return'].rolling(w, min_periods=1).mean()

data = data.iloc[max(windows):]

# ======== 3. DEFINICIÓN DE RÉGIMENES ========
trend_up = data['return'].rolling(30, min_periods=10).mean()
vol_threshold = data['return'].rolling(30, min_periods=10).std().quantile(0.7)

data['trend_category'] = pd.qcut(trend_up, q=3, labels=['bear','range','bull'])
data['regime'] = data['trend_category'].astype(str)
data.loc[data['return'].rolling(30).std() > vol_threshold, 'regime'] = 'volatile'
data.drop(columns=['trend_category'], inplace=True)

# ======== 4. LIMPIEZA ========
feature_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'macd', 'macdsig', 'macdhist',
    'boll_upper', 'boll_middle', 'boll_lower',
    'return', 'log_return'
] + [f'volatility_{w}' for w in windows] + [f'return_{w}' for w in windows]
feature_cols = [f for f in feature_cols if f in data.columns]

data = data.dropna(subset=['regime'] + feature_cols)
X = data[feature_cols]
y = data['regime']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nDistribución de regímenes:")
print(pd.Series(y).value_counts(normalize=True))

tscv = TimeSeriesSplit(n_splits=3)

# ======== 5. OBJETIVO OPTUNA ========
def objective(trial):
    # Hiperparámetros de los 3 modelos
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 500, 2500, step=500),
        'max_depth': trial.suggest_int('rf_max_depth', 10, 100, step=10),
        'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    et_params = {
        'n_estimators': trial.suggest_int('et_n_estimators', 500, 2000, step=500),
        'max_depth': trial.suggest_int('et_max_depth', 10, 100, step=10),
        'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    hgb_params = {
        'max_iter': trial.suggest_int('hgb_max_iter', 100, 1000, step=100),
        'max_depth': trial.suggest_int('hgb_max_depth', 3, 15),
        'learning_rate': trial.suggest_float('hgb_lr', 0.01, 0.3),
        'max_leaf_nodes': trial.suggest_int('hgb_max_leaf_nodes', 10, 50),
        'random_state': 42
    }

    rf = RandomForestClassifier(**rf_params)
    et = ExtraTreesClassifier(**et_params)
    hgb = HistGradientBoostingClassifier(**hgb_params)

    # Ensemble promedio de predicciones
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        if len(np.unique(y_train)) > 1:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        rf.fit(X_train_res, y_train_res)
        et.fit(X_train_res, y_train_res)
        hgb.fit(X_train_res, y_train_res)

        # Promedio de predicciones por probabilidad
        pred_prob = (rf.predict_proba(X_test) + et.predict_proba(X_test) + hgb.predict_proba(X_test)) / 3
        y_pred = np.argmax(pred_prob, axis=1)
        scores.append(balanced_accuracy_score(y_test, y_pred))

    return np.mean(scores)

# ======== 6. OPTUNA STUDY ========
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # subir a 50-100 si quieres más precisión

print("\nBest hyperparameters:", study.best_params)
print("Best Balanced Accuracy:", study.best_value)

# ======== 7. ENTRENAMIENTO FINAL ========
# Extraer los mejores parámetros
best_params = study.best_params

final_rf = RandomForestClassifier(
    n_estimators=best_params['rf_n_estimators'],
    max_depth=best_params['rf_max_depth'],
    max_features=best_params['rf_max_features'],
    min_samples_split=best_params['rf_min_samples_split'],
    min_samples_leaf=best_params['rf_min_samples_leaf'],
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
final_et = ExtraTreesClassifier(
    n_estimators=best_params['et_n_estimators'],
    max_depth=best_params['et_max_depth'],
    max_features=best_params['et_max_features'],
    min_samples_split=best_params['et_min_samples_split'],
    min_samples_leaf=best_params['et_min_samples_leaf'],
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
final_hgb = HistGradientBoostingClassifier(
    max_iter=best_params['hgb_max_iter'],
    max_depth=best_params['hgb_max_depth'],
    learning_rate=best_params['hgb_lr'],
    max_leaf_nodes=best_params['hgb_max_leaf_nodes'],
    random_state=42
)

if len(np.unique(y_encoded)) > 1:
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_encoded)
else:
    X_res, y_res = X, y_encoded

final_rf.fit(X_res, y_res)
final_et.fit(X_res, y_res)
final_hgb.fit(X_res, y_res)

# Guardamos ensemble completo
ensemble_model = {
    'rf': final_rf,
    'et': final_et,
    'hgb': final_hgb,
    'label_encoder': le,
    'features': feature_cols
}

model_folder = 'models/L3'
os.makedirs(model_folder, exist_ok=True)
model_file = os.path.join(model_folder, 'regime_detection_model_ensemble_optuna.pkl')
joblib.dump(ensemble_model, model_file)

print(f"\n✅ Modelo L3 definitivo entrenado y guardado en '{model_file}'")
