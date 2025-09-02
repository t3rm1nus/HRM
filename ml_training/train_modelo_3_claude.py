def create_trading_aligned_target(df: pd.DataFrame, 
                                stop_loss_pct: float = 0.02,    # 2% stop-loss
                                take_profit_pct: float = 0.04,  # 4% take-profit
                                max_hold_periods: int = 60,     # Máximo 10 minutos (60 ticks de 10s)
                                transaction_cost: float = 0.001 # 0.1% costos
                               ) -> pd.Series:
    """
    Crea target alineado con objetivos reales de trading.
    
    Para cada punto, simula:
    1. Abrir posición LONG
    2. Cerrar cuando se alcance stop-loss, take-profit, o tiempo máximo
    3. Calcular P&L neto (incluyendo costos)
    4. Target = 1 si P&L > 0, 0 si P&L <= 0
    
    Esto entrena el modelo para identificar señales REALMENTE rentables.
    """
    closes = df['close'].values
    target = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - max_hold_periods):
        entry_price = closes[i]
        
        # Precios de salida
        stop_price = entry_price * (1 - stop_loss_pct)
        profit_price = entry_price * (1 + take_profit_pct)
        
        # Buscar primer trigger de salida
        exit_price = None
        exit_reason = None
        
        for j in range(i + 1, min(i + max_hold_periods + 1, len(df))):
            current_price = closes[j]
            
            # Check stop-loss
            if current_price <= stop_price:
                exit_price = stop_price
                exit_reason = 'stop_loss'
                break
                
            # Check take-profit
            if current_price >= profit_price:
                exit_price = profit_price
                exit_reason = 'take_profit'
                break
        
        # Si no se activó ni stop ni profit, cerrar a precio de mercado
        if exit_price is None:
            exit_price = closes[min(i + max_hold_periods, len(df) - 1)]
            exit_reason = 'timeout'
        
        # Calcular P&L neto
        raw_return = (exit_price - entry_price) / entry_price
        net_return = raw_return - (2 * transaction_cost)  # Compra + venta
        
        # Target: 1 si rentable, 0 si no
        target[i] = 1 if net_return > 0 else 0
    
    return pd.Series(target, index=df.index)


def create_signal_quality_target(df: pd.DataFrame,
                                signal_strength_periods: int = 10,  # Evaluar señal en próximos 10 períodos
                                min_favorable_ratio: float = 0.6    # 60% períodos favorables = buena señal
                               ) -> pd.Series:
    """
    Target alternativo: Calidad de señal basada en consistencia direccional.
    
    Para cada punto, evalúa si la tendencia futura es consistentemente alcista
    durante los próximos N períodos.
    """
    closes = df['close'].values
    target = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - signal_strength_periods):
        current_price = closes[i]
        
        # Contar períodos alcistas en ventana futura
        favorable_periods = 0
        
        for j in range(i + 1, min(i + signal_strength_periods + 1, len(df))):
            if closes[j] > current_price:
                favorable_periods += 1
        
        # Target = 1 si la mayoría de períodos futuros son favorables
        favorable_ratio = favorable_periods / signal_strength_periods
        target[i] = 1 if favorable_ratio >= min_favorable_ratio else 0
    
    return pd.Series(target, index=df.index)


def create_l1_filter_target(df: pd.DataFrame,
                           l2_signals: pd.Series,        # Señales de L2 (1=buy, 0=sell/neutral)
                           success_threshold: float = 0.01  # 1% ganancia mínima para considerar éxito
                          ) -> pd.Series:
    """
    Target específico para L1: ¿Esta señal de L2 será exitosa?
    
    Simula el caso de uso real:
    1. L2 genera señal BUY en tiempo t
    2. L1 debe decidir si EJECUTAR o RECHAZAR
    3. Target = 1 si ejecutar la señal habría sido rentable
    """
    closes = df['close'].values
    target = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - 20):  # Buffer para evaluación futura
        if l2_signals.iloc[i] == 1:  # Solo evaluar cuando L2 dice "BUY"
            entry_price = closes[i]
            
            # Buscar mejor salida en próximos períodos (ventana realista)
            max_return = -float('inf')
            
            for j in range(i + 1, min(i + 20, len(df))):  # Evaluar próximos 20 períodos
                current_return = (closes[j] - entry_price) / entry_price
                max_return = max(max_return, current_return)
            
            # Target = 1 si la señal habría sido rentable
            target[i] = 1 if max_return >= success_threshold else 0
        
        # Si L2 no genera señal BUY, L1 no necesita decidir (target = 0 por defecto)
    
    return pd.Series(target, index=df.index)


# IMPLEMENTACIÓN EN EL CÓDIGO DE ENTRENAMIENTO
def _make_xy_trading_aligned(df: pd.DataFrame, target_type: str = 'trading_pnl') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reemplaza _make_xy_multiasset() con targets alineados al trading.
    """
    
    if target_type == 'trading_pnl':
        # Target basado en P&L real de trading
        y = create_trading_aligned_target(df)
        
    elif target_type == 'signal_quality':
        # Target basado en calidad de señal
        y = create_signal_quality_target(df)
        
    elif target_type == 'l1_filter':
        # Target específico para L1 (requiere señales de L2)
        # Simular señales de L2 para ejemplo (en producción vendrían de L2)
        rsi = df.get('rsi', pd.Series(50, index=df.index))
        simulated_l2_signals = (rsi < 30).astype(int)  # RSI oversold como proxy
        y = create_l1_filter_target(df, simulated_l2_signals)
        
    else:
        raise ValueError(f"Tipo de target desconocido: {target_type}")
    
    # Features (igual que antes)
    exclude_cols = ['close', 'symbol'] if 'symbol' in df.columns else ['close']
    X = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
    
    # Filtrar datos válidos
    valid = X.notna().all(axis=1) & y.notna() & (y != -1)
    
    return X.loc[valid], y.loc[valid]


# EJEMPLO DE USO EN ENTRENAMIENTO
if __name__ == "__main__":
    # En lugar de usar _make_xy_multiasset(), usar:
    
    # Para modelo general de trading
    X_train, y_train = _make_xy_trading_aligned(train_df, target_type='trading_pnl')
    
    # Para filtro específico de L1
    # X_train, y_train = _make_xy_trading_aligned(train_df, target_type='l1_filter')
    
    print(f"🎯 Target alineado con trading: {y_train.mean():.3f} señales positivas")
    print(f"📊 Distribución: {y_train.value_counts().to_dict()}")