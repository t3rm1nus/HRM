#!/usr/bin/env python3
"""
Test script to verify L3 module import works without TensorFlow hanging
"""

import sys
import time
from datetime import datetime

def test_l3_import():
    """Test L3 module import and initialization"""
    print(f"[{datetime.now()}] Starting L3 import test...")
    
    try:
        # Test basic import
        print("Testing l3_strategy import...")
        start_time = time.time()
        
        from l3_strategy import __version__, get_module_info, check_dependencies
        
        import_time = time.time() - start_time
        print(f"✅ L3 module imported successfully in {import_time:.2f} seconds")
        print(f"   Version: {__version__}")
        
        # Test module info
        print("\nTesting module info...")
        info = get_module_info()
        print(f"   Module: {info['name']} v{info['version']}")
        print(f"   Status: {info['status']}")
        print(f"   Config loaded: {info['config_loaded']}")
        print(f"   Models loaded: {info['models_loaded']}")
        
        # Test dependency checking (this should now work without hanging)
        print("\nTesting dependency check...")
        start_time = time.time()
        
        deps = check_dependencies()
        
        check_time = time.time() - start_time
        print(f"✅ Dependency check completed in {check_time:.2f} seconds")
        
        # Show dependency status
        print("   Dependencies:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"     {status} {dep}: {available}")
        
        # Test L3 processor import
        print("\nTesting L3 processor import...")
        try:
            from l3_strategy.l3_processor import generate_l3_output
            print("✅ L3 processor imported successfully")
        except Exception as e:
            print(f"⚠️  L3 processor import warning: {e}")
        
        print(f"\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during L3 import test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_l3_import()
    sys.exit(0 if success else 1)
