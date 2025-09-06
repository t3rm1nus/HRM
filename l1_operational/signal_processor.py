"""
Signal processor module for converting between different signal types
"""
from typing import Dict, Any, Optional, Union
import pandas as pd
import time
import uuid
from .models import Signal, create_signal
from l2_tactic.models import TacticalSignal
from core.logging import logger

def process_tactical_signal(signal: Union[dict, list, TacticalSignal]) -> Optional[Signal]:
    """
    Convert a tactical signal (dict, list, or TacticalSignal) to a Signal object
    """
    try:
        # Handle list case - take the first signal
        if isinstance(signal, list):
            if not signal:
                logger.warning("Empty signal list received")
                return None
            signal = signal[0]
            
        # Convert dict to TacticalSignal if needed
        if isinstance(signal, dict):
            try:
                # Process numeric values
                signal_data = signal.copy()
                for k in ['strength', 'confidence']:
                    if k in signal_data:
                        signal_data[k] = float(signal_data[k])
                
                # Process features and metadata
                features = signal_data.get('features', {})
                metadata = signal_data.get('metadata', {})
                if isinstance(metadata, dict):
                    # Add numerical metadata to features
                    features.update({
                        k: float(v) for k, v in metadata.items()
                        if isinstance(v, (int, float))
                    })
                signal_data['features'] = features
                
                # Convert timestamp if needed
                if isinstance(signal_data.get('timestamp'), pd.Timestamp):
                    signal_data['timestamp'] = signal_data['timestamp'].timestamp()
                    
                signal = TacticalSignal(**signal_data)
            except Exception as e:
                logger.error(f"❌ Error converting dict to TacticalSignal: {e}")
                return None
        
        # Now create a proper Signal object with all required fields
        signal_dict = {
            'signal_id': str(uuid.uuid4()),
            'strategy_id': 'L2_TACTIC',
            'symbol': signal.symbol,
            'side': signal.side.lower(),
            'qty': 0.0,  # Will be calculated later
            'order_type': getattr(signal, 'type', 'market'),
            'confidence': float(getattr(signal, 'confidence', 0.5)),
            'timestamp': time.time(),
            'technical_indicators': {},
            'strength': float(getattr(signal, 'strength', 0.5)),
            'signal_type': getattr(signal, 'signal_type', 'tactical')
        }
        
        # Extract and process features
        features = {}
        if hasattr(signal, 'features'):
            features = signal.features or {}
        elif isinstance(signal, dict):
            features = signal.get('features', {})
            # Try alternate sources if features is empty
            if not features:
                if 'indicators' in signal:
                    features = signal['indicators']
                elif 'technical_indicators' in signal:
                    features = signal['technical_indicators']
        
        # Ensure features is a dictionary
        if not isinstance(features, dict):
            features = {}
            
        # Add required technical indicators
        signal_dict['technical_indicators'] = {
            'signal_strength': float(getattr(signal, 'strength', 0.5)),
            'rsi': float(features.get('rsi', 50.0)),
            'macd': float(features.get('macd', 0.0)),
            'macd_signal': float(features.get('macd_signal', 0.0)),
            'sma_20': float(features.get('sma_20', 0.0)),
            'sma_50': float(features.get('sma_50', 0.0)),
            'bollinger_upper': float(features.get('bollinger_upper', 0.0)),
            'bollinger_lower': float(features.get('bollinger_lower', 0.0)),
            'vol_zscore': float(features.get('vol_zscore', 0.0))
        }
        
        # Store complete features for AI processing
        signal_dict['features'] = features

        # Handle price - try multiple sources
        price = None
        # Try direct price attribute
        if hasattr(signal, 'price'):
            price = getattr(signal, 'price')
        # Try price in features
        if price is None and features:
            price = features.get('close')
        # Try price in technical indicators
        if price is None and 'close' in signal_dict['technical_indicators']:
            price = signal_dict['technical_indicators']['close']
        # Try market data as last resort
        if price is None and isinstance(signal, dict) and 'market_data' in signal:
            market_data = signal['market_data']
            if isinstance(market_data, dict) and 'close' in market_data:
                price = market_data['close']
                
        signal_dict['price'] = float(price) if price is not None else 0.0  # Default to 0.0 instead of None
        
        # Add timestamp if available
        if hasattr(signal, 'timestamp'):
            if isinstance(signal.timestamp, pd.Timestamp):
                signal_dict['timestamp'] = signal.timestamp.timestamp()
            elif signal.timestamp is not None:
                signal_dict['timestamp'] = float(signal.timestamp)
        
        # Add features as technical indicators
        if hasattr(signal, 'features'):
            features = signal.features
            if isinstance(features, dict):
                signal_dict['technical_indicators'].update(
                    {k: float(v) for k, v in features.items() 
                     if isinstance(v, (int, float))}
                )
        
        # Ensure all required fields are present
        if signal_dict.get('price') is None:
            signal_dict['price'] = 0.0  # Default price
            logger.warning(f"No price found for {signal_dict['symbol']}, using default 0.0")
        
        # Create final Signal object
        signal_obj = create_signal(**signal_dict)
        logger.debug(f"Created signal: {signal_dict['symbol']} {signal_dict['side']} with strength {signal_dict['technical_indicators'].get('signal_strength', 0.0)}")
        return signal_obj
        
    except Exception as e:
        logger.error(f"❌ Error processing tactical signal: {e}")
        return None
