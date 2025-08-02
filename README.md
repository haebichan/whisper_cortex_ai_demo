# Real-Time Speech-to-Text with OpenAI Whisper

A modular Streamlit application that provides real-time speech-to-text transcription using OpenAI's Whisper model and WebRTC for audio streaming.

## Features

- **Real-time transcription**: Convert speech to text in real-time using WebRTC
- **Multiple Whisper models**: Choose from tiny, base, small, medium, or large models
- **Language support**: Auto-detection or manual language selection
- **Two modes**: Audio-only or video+audio streaming
- **Configurable settings**: Adjustable chunk duration and audio processing parameters
- **Clean architecture**: Modular design for easy maintenance and extension

## Architecture

The application is organized into several modules for clean separation of concerns:

```
speech_to_text/
├── __init__.py                 # Package initialization
├── main.py                     # Main application entry point
├── config.py                   # Configuration management
├── network.py                  # WebRTC and network configuration
├── audio_processor.py          # Audio processing utilities
├── whisper_service.py          # Whisper model management
├── ui_components.py            # UI components and interactions
├── webrtc_handlers.py          # WebRTC stream handlers
└── logger_setup.py             # Logging configuration
```

### Module Overview

- **config.py**: Centralized configuration management with dataclasses for different components
- **network.py**: Handles ICE server configuration and WebRTC networking
- **audio_processor.py**: Manages audio frame accumulation and processing
- **whisper_service.py**: Encapsulates Whisper model loading and transcription logic
- **ui_components.py**: Reusable UI components and user interaction handling
- **webrtc_handlers.py**: Separate handlers for audio-only and video+audio modes
- **logger_setup.py**: Centralized logging configuration
- **main.py**: Application orchestration and main entry point

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-to-text-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Twilio credentials for TURN server support:
```bash
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
```

## Usage

### Running the Application

```bash
streamlit run main.py
```

Or import and run programmatically:

```python
from speech_to_text import main
main()
```

### Configuration Options

- **Model Size**: Choose between tiny (fastest) to large (most accurate)
- **Chunk Duration**: Adjust how often transcription occurs (1-10 seconds)
- **Language**: Select specific language or use auto-detection
- **Mode**: Choose between audio-only or video+audio streaming

## Configuration

The application uses several configuration classes:

### AudioConfig
- `chunk_duration`: How often to transcribe (default: 3 seconds)
- `sample_rate`: Audio sample rate (default: 16000 Hz)
- `channels`: Number of audio channels (default: 1)

### WhisperConfig
- `model_size`: Whisper model size (default: "base")
- `language`: Language code or None for auto-detection
- `fp16`: Use half-precision floating point (default: True)

### TwilioConfig
- Automatically loads from environment variables
- Falls back to Google STUN server if not configured

## Extending the Application

### Adding New Audio Processing Features

Extend the `AudioProcessor` class in `audio_processor.py`:

```python
class CustomAudioProcessor(AudioProcessor):
    def custom_processing_method(self):
        # Your custom logic here
        pass
```

### Adding New UI Components

Add methods to the `UIComponents` class in `ui_components.py`:

```python
class UIComponents:
    @staticmethod
    def render_custom_component():
        # Your custom UI component
        pass
```

### Custom Configuration

Create custom configuration classes following the pattern in `config.py`:

```python
@dataclass
class CustomConfig:
    custom_parameter: str = "default_value"
```

## Development

### Running in Debug Mode

```bash
export DEBUG=true
streamlit run main.py
```

### Testing

The modular architecture makes it easy to unit test individual components:

```python
# Example: Testing audio processor
from speech_to_text.audio_processor import AudioProcessor
from speech_to_text.config import AudioConfig

config = AudioConfig(chunk_duration=2)
processor = AudioProcessor(config)
# ... your tests
```

## Dependencies

- **streamlit**: Web application framework
- **streamlit-webrtc**: WebRTC integration for Streamlit
- **openai-whisper**: Speech recognition model
- **av**: Audio/video processing
- **pydub**: Audio manipulation
- **twilio**: TURN server support
- **torch**: PyTorch for ML inference
- **numpy**: Numerical computing

## Environment Variables

- `TWILIO_ACCOUNT_SID`: Twilio account SID (optional)
- `TWILIO_AUTH_TOKEN`: Twilio auth token (optional)
- `DEBUG`: Enable debug logging (optional)

## Troubleshooting

### Common Issues

1. **Audio not working**: Ensure microphone permissions are granted in your browser
2. **Model loading errors**: Check that you have sufficient RAM for the selected model
3. **Connection issues**: Verify Twilio credentials or check network configuration
4. **Performance issues**: Try using a smaller Whisper model (tiny or base)

### Performance Optimization

- Use smaller Whisper models for faster transcription
- Adjust chunk duration based on your needs
- Enable GPU acceleration if available (automatic with CUDA)

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]