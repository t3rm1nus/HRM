#!/usr/bin/env python3
"""Test logging functionality to ensure it works after DB fix."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the logging system
print("Testing logging system...")

try:
    from core.logging import info, warning, error, setup_logger
    print("✓ Logging modules imported successfully")

    # Ensure setup is called
    logger = setup_logger()
    print("✓ Logger setup completed")

    # Test basic logging
    info("Test message: Logging system is working!", module="test_logging", extra={"test": "123"})
    warning("Test warning message", module="test_logging")
    error("Test error message", module="test_logging")

    print("✓ All test messages sent successfully")

    # Check database after logs
    import sqlite3
    conn = sqlite3.connect('logs/logs.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM logs")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"✓ Database contains {count} log entries")

    if count > 0:
        print("🎉 Logging system is fully functional!")
    else:
        print("❌ No log entries found in database")

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
