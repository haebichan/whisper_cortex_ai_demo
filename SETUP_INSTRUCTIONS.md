# Directory Structure Setup Guide

## Step 1: Create the Directory Structure

Create the following directory structure in your project folder:

```
your_project_folder/
├── app.py                          # Main entry point (simple runner)
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── SETUP_INSTRUCTIONS.md          # This file
└── speech_to_text/                # Main package directory
    ├── __init__.py                 # Package initialization
    ├── main.py                     # Main application logic
    ├── config.py                   # Configuration management
    ├── network.py                  # Network/WebRTC utilities
    ├── audio_processor.py          # Audio processing
    ├── whisper_service.py          # Whisper model management
    ├── ui_components.py            # UI components
    ├── webrtc_handlers.py          # WebRTC stream handlers
    └── logger_setup.py             # Logging setup
```

## Step 2: Create the Files

1. **Create the `speech_to_text` directory**:
   ```bash
   mkdir speech_to_text
   ```

2. **Copy each of the provided files into the `speech_to_text` directory**:
   - `__init__.py`
   - `main.py`
   - `config.py`
   - `network.py`
   - `audio_processor.py`
   - `whisper_service.py`
   - `ui_components.py`
   - `webrtc_handlers.py`
   - `logger_setup.py`

3. **Create the simple `app.py` runner** in the root directory (see the app.py artifact)

4. **Copy `requirements.txt`** to the root directory

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Application

```bash
streamlit run app.py
```

## Important Notes:

- The `speech_to_text` folder acts as a Python package
- The `__init__.py` file makes it importable
- All the modular files go inside the `speech_to_text` folder
- The simple `app.py` stays in the root and imports from the package

## If you get import errors:

Make sure your directory structure matches exactly what's shown above, and that you have an `__init__.py` file in the `speech_to_text` directory.