# test_flower_toml.py
import subprocess
import sys

# Запустим flower с максимальным выводом ошибок
result = subprocess.run([
    sys.executable, "-m", "flwr", "run", 
    "--verbose", "debug"
], capture_output=True, text=True)

print("Return code:", result.returncode)
print("STDOUT:")
print(result.stdout)
print("STDERR:") 
print(result.stderr)