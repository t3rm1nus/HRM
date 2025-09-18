# -*- coding: utf-8 -*-
# Signal validation utilities

import pandas as pd
import numpy as np
from typing import List, Any, Optional
from core.logging import logger
from l2_tactic.models import TacticalSignal

def validate_tactical_signal(signal: Any) -> Optional[TacticalSignal]:
    """Validate and fix TacticalSignal object"""
    try:
        if signal is None:
            return None
            
        if not isinstance(signal, TacticalSignal):
            logger.warning(f"⚠️ Signal is not TacticalSignal type: {type(signal)}")
            return None
        
        # Ensure required attributes exist
        required_attrs = ['symbol', 'side', 'strength', 'confidence']
        for attr in required_attrs:
            if not hasattr(signal, attr):
                logger.error(f"❌ Signal missing required attribute: {attr}")
                return None
                
        # Validate side/action attribute
        if hasattr(signal, 'side'):
            valid_sides = ['buy', 'sell', 'hold']
            if signal.side not in valid_sides:
                logger.warning(f"⚠️ Invalid signal side: {signal.side}, defaulting to 'hold'")
                signal.side = 'hold'
        
        # Ensure action attribute exists (for backward compatibility)
        if not hasattr(signal, 'action'):
            signal.action = signal.side
            
        # Validate numeric values
        if pd.isna(signal.strength) or not isinstance(signal.strength, (int, float)):
            signal.strength = 0.5
            
        if pd.isna(signal.confidence) or not isinstance(signal.confidence, (int, float)):
            signal.confidence = 0.5
            
        # Ensure timestamp is proper format
        if not hasattr(signal, 'timestamp') or signal.timestamp is None:
            signal.timestamp = pd.Timestamp.utcnow()
        elif not isinstance(signal.timestamp, pd.Timestamp):
            try:
                signal.timestamp = pd.to_datetime(signal.timestamp)
            except Exception:
                signal.timestamp = pd.Timestamp.utcnow()
                
        # Ensure features is a dict
        if not hasattr(signal, 'features') or not isinstance(signal.features, dict):
            signal.features = {}
            
        # Ensure metadata is a dict
        if not hasattr(signal, 'metadata') or not isinstance(signal.metadata, dict):
            signal.metadata = {}
            
        return signal
        
    except Exception as e:
        logger.error(f"❌ Error validating signal: {e}")
        return None

def validate_signal_list(signals: Any) -> List[TacticalSignal]:
    """Validate and clean a list of signals"""
    try:
        if signals is None:
            return []
            
        if not isinstance(signals, (list, tuple)):
            logger.warning(f"⚠️ Signals is not a list: {type(signals)}")
            if hasattr(signals, '__iter__'):
                try:
                    signals = list(signals)
                except Exception:
                    return []
            else:
                return []
        
        valid_signals = []
        for i, signal in enumerate(signals):
            validated = validate_tactical_signal(signal)
            if validated is not None:
                valid_signals.append(validated)
            else:
                logger.warning(f"⚠️ Removed invalid signal at index {i}")
                
        logger.info(f"✅ Validated {len(valid_signals)} out of {len(signals)} signals")
        return valid_signals
        
    except Exception as e:
        logger.error(f"❌ Error validating signal list: {e}")
        return []

def create_fallback_signal(symbol: str, reason: str = "fallback") -> TacticalSignal:
    """Create a safe fallback signal"""
    return TacticalSignal(
        symbol=symbol,
        side='hold',
        strength=0.1,
        confidence=0.1,
        signal_type='fallback',
        source=f'fallback_{reason}',
        features={},
        timestamp=pd.Timestamp.utcnow(),
        metadata={'reason': reason}
    )
