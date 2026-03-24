import subprocess
import sys

if __name__ == '__main__':
    exit(subprocess.call([sys.executable, 'runner.py', '--benchmark', 'fused']))
