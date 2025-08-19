import os

# Definir estructura de carpetas y archivos
structure = {
    "HMR/docs": [],
    "HMR/core": ["logging.py", "scheduler.py", "utils.py"],
    "HMR/core/config": [],
    "HMR/comms": ["message_bus.py", "schemas.py"],
    "HMR/comms/adapters": [],
    "HMR/l4_meta": ["drift_detector.py", "strategy_selector.py", "portfolio_allocator.py", "__init__.py"],
    "HMR/l3_strategy": ["regime_classifier.py", "universe_filter.py", "exposure_manager.py", "__init__.py"],
    "HMR/l2_tactic": ["signal_generator.py", "position_sizer.py", "risk_controls.py", "__init__.py"],
    "HMR/l1_operational": ["order_manager.py", "execution_algos.py", "realtime_risk.py", "__init__.py"],
    "HMR/data": ["loaders.py", "__init__.py"],
    "HMR/data/connectors": [],
    "HMR/data/storage": [],
    "HMR/risk": ["limits.py", "var_es.py", "drawdown.py", "__init__.py"],
    "HMR/monitoring": ["alerts.py", "telemetry.py", "__init__.py"],
    "HMR/monitoring/dashboards": [],
    "HMR/tests": [],
}

# Crear carpetas y archivos
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            with open(path, "w") as fp:
                fp.write("# " + f + "\n")

# Crear main.py en la raíz
main_path = "HMR/main.py"
if not os.path.exists(main_path):
    with open(main_path, "w") as f:
        f.write("# Orquestador central\n\nif __name__ == '__main__':\n    print('HMR system running...')\n")

print("Estructura de proyecto HMR creada exitosamente.")
