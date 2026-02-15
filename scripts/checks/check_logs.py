import os
import json
import csv

# Check logging files
print("Checking logging files...")

log_files = {
    'JSON logs': 'logs/events.json',
    'CSV logs': 'logs/events.csv'
}

all_checks_passed = True

for file_type, file_path in log_files.items():
    print(f"\nChecking {file_type}...")

    if os.path.exists(file_path):
        try:
            file_size = os.path.getsize(file_path)
            print(f"✓ {file_type} file exists ({file_size} bytes)")

            # Quick validation
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Check if it's valid JSON Lines format
                        line_count = 0
                        for line in f:
                            if line.strip():
                                json.loads(line.strip())
                                line_count += 1
                                if line_count >= 10:  # Check first 10 lines
                                    break
                    print(f"✓ JSON format valid (checked {min(line_count, 10)} entries)")
                except Exception as e:
                    print(f"✗ JSON format error: {e}")
                    all_checks_passed = False

            elif file_path.endswith('.csv'):
                try:
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        print(f"✓ CSV format valid (columns: {len(header)} - {header[:3]}...)")
                except Exception as e:
                    print(f"✗ CSV format error: {e}")
                    all_checks_passed = False

        except Exception as e:
            print(f"✗ Error accessing {file_type}: {e}")
            all_checks_passed = False
    else:
        print(f"⚠ {file_type} file does not exist yet")

print(f"\n{'='*50}")
if all_checks_passed:
    print("✅ All logging file checks passed!")
else:
    print("❌ Some logging file checks failed.")
print(f"{'='*50}")
