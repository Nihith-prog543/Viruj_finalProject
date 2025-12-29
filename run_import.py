import subprocess
import sys
import os

# Change to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Run the import script
result = subprocess.run([sys.executable, "import_manufacturers.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)
sys.exit(result.returncode)


