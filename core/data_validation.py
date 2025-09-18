# -*- coding: utf-8 -*-
# Data validation utilities

from typing import Any


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
