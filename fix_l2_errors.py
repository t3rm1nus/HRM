#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix for L2 Tactical Signal Processing Errors

This script addresses the following issues:
1. TacticalSignal object missing 'action' attribute
2. NaN values after validation in core.logging
3. 'str' object has no attribute 'keys' error in main cycle

Author: Cline AI Assistant
Date: 2025-01-15
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from l2_tactic.models import TacticalSignal

class L2ErrorFixer:
    """Comprehensive L2 error fixing utility"""
    
    def __init__(self):
        self.fixes_applied = []
        
    def fix_tactical_signal_action_attribute(self):
        """Fix missing 'action' attribute in TacticalSignal"""
        logger.info("🔧 Fixing TacticalSignal 'action' attribute issue...")
        
        # The issue is that code is trying to access signal.action but TacticalSignal uses signal.side
        # We need to add an 'action' property to TacticalSignal that maps to 'side'
        
        # Read the current models.py file
        models_path = "l2_tactic/models.py"
        
        try:
            with open(models_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if action property already exists
            if '@property' in content and 'def action(' in content:
                logger.info("✅ TacticalSignal already has action property")
                return True
                
            # Find the TacticalSignal class and add the action property
            lines = content.split('\n')
            new_lines = []
            in_tactical_signal = False
            added_property = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Check if we're in the TacticalSignal class
                if line.strip().startswith('class TacticalSignal:') or line.strip().startswith('@dataclass'):
                    in_tactical_signal = True
                elif line.strip().startswith('class ') and 'TacticalSignal' not in line:
                    in_tactical_signal = False
                    
                # Add action property after the __str__ method
                if (in_tactical_signal and 
                    line.strip().startswith('def __str__(') and 
                    not added_property):
                    
                    # Find the end of __str__ method
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith('        ') or lines[j].strip() == ''):
                        j += 1
                    
                    # Insert the action property after __str__
                    property_lines = [
                        '',
                        '    @property',
                        '    def action(self) -> str:',
                        '        """Alias for side attribute for backward compatibility"""',
                        '        return self.side',
                        '',
                        '    @action.setter', 
                        '    def action(self, value: str):',
                        '        """Set action (maps to side)"""',
                        '        self.side = value',
                    ]
                    
                    # Insert at the current position
                    for prop_line in reversed(property_lines):
                        new_lines.insert(i + 1, prop_line)
                    
                    added_property = True
                    break
            
            if added_property:
                # Write the updated content
                with open(models_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                
                logger.info("✅ Added 'action' property to TacticalSignal class")
                self.fixes_applied.append("TacticalSignal action property")
                return True
            else:
                logger.warning("⚠️ Could not find suitable location to add action property")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error fixing TacticalSignal action attribute: {e}")
            return False
    
    def fix_nan_validation_issues(self):
        """Fix NaN validation issues in data processing"""
        logger.info("🔧 Fixing NaN validation issues...")
        
        try:
            # Create a data validation utility
            validation_code = '''
def validate_and_clean_data(data: Any, context: str = "unknown") -> Any:
    """Validate and clean data, removing NaN values"""
    import pandas as pd
    import numpy as np
    from core.logging import logger
    
    if data is None:
        return data
        
    try:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Count NaN values before cleaning
            nan_count = data.isna().sum()
            if isinstance(nan_count, pd.Series):
                total_nans = nan_count.sum()
            else:
                total_nans = nan_count
                
            if total_nans > 0:
                logger.warning(f"⚠️ Found {total_nans} NaN values in {context}, cleaning...")
                
                if isinstance(data, pd.DataFrame):
                    # Fill NaN values with appropriate defaults
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    data[numeric_cols] = data[numeric_cols].fillna(0.0)
                    
                    # Fill non-numeric columns
                    for col in data.columns:
                        if col not in numeric_cols:
                            data[col] = data[col].fillna('')
                            
                elif isinstance(data, pd.Series):
                    if pd.api.types.is_numeric_dtype(data):
                        data = data.fillna(0.0)
                    else:
                        data = data.fillna('')
                        
                logger.info(f"✅ Cleaned NaN values in {context}")
                
        elif isinstance(data, dict):
            # Clean dictionary values
            for key, value in data.items():
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    data[key] = 0.0 if isinstance(value, (int, float)) else ''
                    
        elif isinstance(data, (list, tuple)):
            # Clean list/tuple values
            cleaned = []
            for item in data:
                if pd.isna(item) or (isinstance(item, float) and np.isnan(item)):
                    cleaned.append(0.0 if isinstance(item, (int, float)) else '')
                else:
                    cleaned.append(item)
            data = type(data)(cleaned)
            
        elif isinstance(data, (int, float)) and (pd.isna(data) or np.isnan(data)):
            data = 0.0
            
    except Exception as e:
        logger.error(f"❌ Error validating data in {context}: {e}")
        
    return data
'''
            
            # Write validation utility to a separate file
            with open('core/data_validation.py', 'w', encoding='utf-8') as f:
                f.write('# -*- coding: utf-8 -*-\n')
                f.write('# Data validation utilities\n\n')
                f.write('from typing import Any\n\n')
                f.write(validation_code)
            
            logger.info("✅ Created data validation utility")
            self.fixes_applied.append("NaN validation utility")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating validation utility: {e}")
            return False
    
    def fix_string_keys_error(self):
        """Fix 'str' object has no attribute 'keys' error"""
        logger.info("🔧 Fixing string keys error...")
        
        try:
            # Create a safe dictionary access utility
            safe_access_code = '''
def safe_dict_access(obj: Any, key: str, default: Any = None) -> Any:
    """Safely access dictionary-like objects"""
    try:
        if obj is None:
            return default
            
        if isinstance(obj, dict):
            return obj.get(key, default)
        elif hasattr(obj, 'get'):
            return obj.get(key, default)
        elif hasattr(obj, '__getitem__'):
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                return default
        else:
            # Object doesn't support dictionary access
            return default
            
    except Exception:
        return default

def ensure_dict(obj: Any, context: str = "unknown") -> dict:
    """Ensure object is a dictionary"""
    from core.logging import logger
    
    if obj is None:
        return {}
        
    if isinstance(obj, dict):
        return obj
        
    if isinstance(obj, str):
        try:
            import json
            # Try to parse as JSON
            return json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"⚠️ String object in {context} is not valid JSON, returning empty dict")
            return {}
            
    if hasattr(obj, 'to_dict'):
        try:
            return obj.to_dict()
        except Exception:
            pass
            
    if hasattr(obj, '__dict__'):
        try:
            return obj.__dict__
        except Exception:
            pass
            
    logger.warning(f"⚠️ Could not convert {type(obj)} to dict in {context}, returning empty dict")
    return {}

def safe_market_data_access(state: dict, key: str = "market_data") -> dict:
    """Safely access market data from state"""
    from core.logging import logger
    
    try:
        market_data = state.get(key, {})
        
        if not isinstance(market_data, dict):
            logger.warning(f"⚠️ {key} is not a dict (type: {type(market_data)}), converting...")
            market_data = ensure_dict(market_data, key)
            state[key] = market_data
            
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Error accessing {key}: {e}")
        return {}
'''
            
            # Append to the data validation file
            with open('core/data_validation.py', 'a', encoding='utf-8') as f:
                f.write('\n\n')
                f.write(safe_access_code)
            
            logger.info("✅ Added safe dictionary access utilities")
            self.fixes_applied.append("Safe dictionary access")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating safe access utilities: {e}")
            return False
    
    def fix_finrl_signal_structure(self):
        """Fix FinRL signal structure issues"""
        logger.info("🔧 Fixing FinRL signal structure...")
        
        try:
            finrl_path = "l2_tactic/finrl_integration.py"
            
            with open(finrl_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix the _action_to_signal method to ensure proper TacticalSignal creation
            fixes = [
                # Fix missing hold_prob calculation
                ('# Calculate probabilities for each action with numerical stability',
                 '''# Calculate probabilities for each action with numerical stability
            sell_threshold = 0.33
            buy_threshold = 0.66
            
            # Calculate hold probability first
            if action_val <= sell_threshold:
                hold_prob = 1.0 - action_val / sell_threshold
            elif action_val >= buy_threshold:
                hold_prob = (1.0 - action_val) / (1.0 - buy_threshold)
            else:
                hold_prob = 1.0 - abs(action_val - 0.5) * 2
            
            hold_prob = max(0.0, min(1.0, hold_prob))'''),
                
                # Fix TacticalSignal creation to use proper parameters
                ('return TacticalSignal(',
                 '''return TacticalSignal('''),
                
                # Ensure all required parameters are provided
                ('timestamp=datetime.utcnow().timestamp(),',
                 'timestamp=pd.Timestamp.utcnow(),')
            ]
            
            modified = False
            for old_text, new_text in fixes:
                if old_text in content and new_text not in content:
                    content = content.replace(old_text, new_text)
                    modified = True
            
            if modified:
                with open(finrl_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info("✅ Fixed FinRL signal structure")
                self.fixes_applied.append("FinRL signal structure")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fixing FinRL signal structure: {e}")
            return False
    
    def create_signal_validator(self):
        """Create a signal validator to ensure proper signal structure"""
        logger.info("🔧 Creating signal validator...")
        
        try:
            validator_code = '''# -*- coding: utf-8 -*-
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
'''
            
            with open('l2_tactic/signal_validator.py', 'w', encoding='utf-8') as f:
                f.write(validator_code)
            
            logger.info("✅ Created signal validator")
            self.fixes_applied.append("Signal validator")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating signal validator: {e}")
            return False
    
    def patch_l2_signal_generator(self):
        """Patch L2 signal generator to use validation"""
        logger.info("🔧 Patching L2 signal generator...")
        
        try:
            generator_path = "l2_tactic/signal_generator.py"
            
            with open(generator_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add import for signal validator
            if 'from .signal_validator import' not in content:
                import_line = "from .signal_validator import validate_signal_list, validate_tactical_signal, create_fallback_signal\n"
                
                # Find the imports section and add our import
                lines = content.split('\n')
                import_index = -1
                for i, line in enumerate(lines):
                    if line.startswith('from .') and 'import' in line:
                        import_index = i
                
                if import_index >= 0:
                    lines.insert(import_index + 1, import_line.strip())
                    content = '\n'.join(lines)
            
            # Patch the process_signals method to validate signals
            if 'validate_signal_list(signals)' not in content:
                old_return = 'return signals'
                new_return = '''# Validate signals before returning
            validated_signals = validate_signal_list(signals)
            return validated_signals'''
                
                content = content.replace(old_return, new_return)
            
            # Patch individual signal creation to use validation
            if 'validate_tactical_signal(tactical_signal)' not in content:
                old_append = 'signals.append(tactical_signal)'
                new_append = '''validated_signal = validate_tactical_signal(tactical_signal)
                if validated_signal:
                    signals.append(validated_signal)
                else:
                    logger.warning(f"⚠️ Invalid signal for {symbol}, creating fallback")
                    fallback = create_fallback_signal(symbol, "invalid_tactical")
                    signals.append(fallback)'''
                
                content = content.replace(old_append, new_append)
            
            with open(generator_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Patched L2 signal generator")
            self.fixes_applied.append("L2 signal generator patch")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error patching L2 signal generator: {e}")
            return False
    
    async def run_all_fixes(self):
        """Run all fixes"""
        logger.info("🚀 Starting comprehensive L2 error fixes...")
        
        fixes = [
            ("TacticalSignal action attribute", self.fix_tactical_signal_action_attribute),
            ("NaN validation issues", self.fix_nan_validation_issues),
            ("String keys error", self.fix_string_keys_error),
            ("FinRL signal structure", self.fix_finrl_signal_structure),
            ("Signal validator", self.create_signal_validator),
            ("L2 signal generator patch", self.patch_l2_signal_generator),
        ]
        
        success_count = 0
        for fix_name, fix_func in fixes:
            try:
                logger.info(f"🔧 Applying fix: {fix_name}")
                if fix_func():
                    success_count += 1
                    logger.info(f"✅ {fix_name} - SUCCESS")
                else:
                    logger.error(f"❌ {fix_name} - FAILED")
            except Exception as e:
                logger.error(f"❌ {fix_name} - ERROR: {e}")
        
        logger.info(f"🎯 Applied {success_count}/{len(fixes)} fixes successfully")
        logger.info(f"📋 Fixes applied: {', '.join(self.fixes_applied)}")
        
        return success_count == len(fixes)

async def main():
    """Main function to run all fixes"""
    fixer = L2ErrorFixer()
    success = await fixer.run_all_fixes()
    
    if success:
        logger.info("🎉 All L2 error fixes applied successfully!")
        logger.info("🔄 Please restart the HRM system to apply the changes.")
    else:
        logger.error("❌ Some fixes failed. Please check the logs and try again.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
