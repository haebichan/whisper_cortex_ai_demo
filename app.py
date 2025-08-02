#!/usr/bin/env python3
"""
Streamlit app runner for the Speech-to-Text application.
This file serves as the entry point when running with `streamlit run app.py`
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main application
from speech_to_text import main

if __name__ == "__main__":
    main()