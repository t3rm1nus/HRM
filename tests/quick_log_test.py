#!/usr/bin/env python3
"""Quick test of logging fix."""

import sys
import os

# Disable TF warnings for quick test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('.')

# Quick import test
print("Testing logging import and basic usage...")

from core.logging import info, setup_logger
import sqlite3

# Setup
logger = setup_logger()

# Log something simple
info("Quick logging test", module="quick_test")

# Check database
conn = sqlite3.connect('logs/logs.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM logs WHERE message LIKE '%Quick logging test%'")
count = cursor.fetchone()[0]
conn.close()

print(f"✓ Logged entry found in database: {count}")
print("✅ Logging fix successful!")
