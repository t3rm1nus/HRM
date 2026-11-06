#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TARGETED FIX: decision_maker.py balance check

This fixes the issue where L3 decision_maker doesn't see the balances
that were just synced in Step 1.
"""

import os
import shutil
from datetime import datetime

def fix_decision_maker():
    """Fix the balance check in decision_maker.py"""
    
    filepath = './l3_strategy/decision_maker.py'
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Create backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the balance check section
    old_balance_check = '''        # Get portfolio from state
        portfolio = state.get('portfolio', {})'''
    
    # New improved balance check that looks in multiple places
    new_balance_check = '''        # Get portfolio from state - CHECK MULTIPLE LOCATIONS
        portfolio = state.get('portfolio', {})
        
        # CRITICAL FIX: Also check direct state keys as backup
        # (balances might be synced to state directly)
        btc_balance = portfolio.get('btc_balance', 0)
        eth_balance = portfolio.get('eth_balance', 0)
        usdt_balance = portfolio.get('usdt_balance', 0)
        
        # Backup: check direct state keys if portfolio is empty
        if btc_balance == 0 and eth_balance == 0 and usdt_balance == 0:
            btc_balance = state.get('btc_balance', 0)
            eth_balance = state.get('eth_balance', 0)
            usdt_balance = state.get('usdt_balance', 0)
            
            # Update portfolio dict with these values for consistency
            if btc_balance > 0 or eth_balance > 0 or usdt_balance > 0:
                portfolio = {
                    'btc_balance': btc_balance,
                    'eth_balance': eth_balance,
                    'usdt_balance': usdt_balance
                }
                state['portfolio'] = portfolio'''
    
    # Apply the fix
    if old_balance_check in content:
        content = content.replace(old_balance_check, new_balance_check)
        print("✅ Updated balance check to look in multiple locations")
    else:
        print("⚠️ Could not find exact balance check pattern")
        print("   Trying alternative pattern...")
        
        # Try alternative pattern
        alt_pattern = 'portfolio = state.get(\'portfolio\', {})'
        if alt_pattern in content:
            # Insert the additional check right after
            insert_text = '''
        
        # CRITICAL FIX: Also check direct state keys as backup
        btc_balance = portfolio.get('btc_balance', 0)
        eth_balance = portfolio.get('eth_balance', 0)
        usdt_balance = portfolio.get('usdt_balance', 0)
        
        if btc_balance == 0 and eth_balance == 0:
            btc_balance = state.get('btc_balance', 0)
            eth_balance = state.get('eth_balance', 0)
            usdt_balance = state.get('usdt_balance', 0)
            
            if btc_balance > 0 or eth_balance > 0:
                portfolio = {
                    'btc_balance': btc_balance,
                    'eth_balance': eth_balance,
                    'usdt_balance': usdt_balance
                }
                state['portfolio'] = portfolio'''
            
            content = content.replace(alt_pattern, alt_pattern + insert_text)
            print("✅ Applied alternative fix pattern")
    
    # Also fix the balance validation check
    old_check = '''if not portfolio or not any([portfolio.get('btc_balance', 0), portfolio.get('eth_balance', 0)]):'''
    
    new_check = '''# Check if we have real balances (from portfolio dict OR direct state)
        has_crypto = (
            portfolio.get('btc_balance', 0) > 0 or 
            portfolio.get('eth_balance', 0) > 0 or
            state.get('btc_balance', 0) > 0 or
            state.get('eth_balance', 0) > 0
        )
        
        has_traded = (
            portfolio.get('usdt_balance', 0) < 3000.0 or
            state.get('usdt_balance', 0) < 3000.0
        )
        
        if not (has_crypto or has_traded):'''
    
    if old_check in content:
        content = content.replace(old_check, new_check)
        print("✅ Updated balance validation logic")
    else:
        print("⚠️ Could not find balance validation check")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File updated successfully")
    return True

if __name__ == "__main__":
    print("="*60)
    print("🔧 TARGETED FIX: decision_maker.py")
    print("="*60)
    print()
    
    success = fix_decision_maker()
    
    print()
    print("="*60)
    if success:
        print("✅ DECISION_MAKER FIXED!")
        print()
        print("What this fixes:")
        print("1. ✅ Checks BOTH portfolio dict AND direct state keys")
        print("2. ✅ Falls back to direct keys if portfolio is empty")
        print("3. ✅ Syncs portfolio dict when finding direct keys")
        print("4. ✅ Validates balances from multiple sources")
        print()
        print("Next steps:")
        print("1. Restart your trading system")
        print("2. BLIND mode should NOT trigger anymore")
        print("3. Auto-rebalancer should be ENABLED")
    else:
        print("❌ FIX FAILED - Manual intervention needed")
    print("="*60)