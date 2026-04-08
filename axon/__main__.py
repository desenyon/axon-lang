"""Allow running Axon as `python -m axon`."""
import sys
import os

# Add parent directory for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.main import main

if __name__ == "__main__":
    main()
