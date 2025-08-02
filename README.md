# 🎤 Voice-Powered Cortex Search

A comprehensive Streamlit application that combines real-time speech-to-text transcription with intelligent document search and AI-powered responses. Speak your questions and get instant, accurate answers from your knowledge base!

## 🌟 Features

- **🎤 Real-time Speech Recognition**: Convert speech to text using OpenAI's Whisper models
- **🔍 Intelligent Document Search**: Search your knowledge base using Snowflake Cortex Search
- **🤖 AI-Powered Responses**: Get summarized, contextual answers using LLMs
- **💬 Interactive Chat Interface**: Conversation-style interface with chat history
- **🎛️ Flexible Configuration**: Customizable audio settings, model selection, and search parameters
- **🌐 Multiple Input Modes**: Voice input OR manual text input
- **📹 Video Support**: Optional video chat mode with audio transcription

## 🏗️ Architecture

The application is built with a clean, modular architecture:

```
speech_to_text/
├── __init__.py                 # Package initialization
├── main.py                     # Main application orchestration
├── config.py                   # Configuration management
├── network.py                  # WebRTC and network configuration
├── audio_processor.py          # Audio frame processing
├── whisper_service.py          # Whisper model management
├── cortex_service.py           # Cortex Search integration ⭐ NEW
├── ui_components.py            # Enhanced UI components
├── webrtc_handlers.py          # WebRTC stream handlers with search
└── logger_setup.py             # Logging configuration
```

### 🔄 Workflow

1. **🎤 Voice Input**: User speaks into microphone
2. **📝 Transcription**: Whisper converts speech to text
3. **🔍 Search**: Query is sent to Cortex Search service
4. **📚 Retrieval**: Relevant documents are found and retrieved
5. **🤖 AI Processing**: LLM summarizes findings into a clear answer
6. **💬 Response**: Answer is displayed in chat interface

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd voice-cortex-search
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Snowflake environment** (required for Cortex Search):
   - Ensure you have access to a Snowflake account with Cortex Search enabled
   - Set up your Cortex Search service in Snowflake
   - Note your search service name (default: `haebi_cortex_search_service`)

4. **(Optional) Configure Twilio for better WebRTC**:
```bash
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
```

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

### Configuration Options

#### 🎤 Audio Settings
- **Model Size**: Choose Whisper model (tiny → large for speed → accuracy trade-off)
- **Chunk Duration**: How often to process audio (1-10 seconds)
- **Language**: Specific language or auto-detection

#### 🔍 Cortex Search Settings
- **Search Service Name**: Your Cortex Search service identifier
- **Document Chunk Limit**: Number of document chunks to retrieve (1-10)
- **Auto-search**: Automatically search when speech is transcribed

#### 🎛️ Interface Modes
- **Sound only**: Audio-only transcription and search
- **With video**: Video chat interface with audio processing

### 💡 How to Use

1. **Start the application** and ensure Cortex Search shows "Connected" ✅
2. **Click "START"** on the WebRTC component
3. **Speak your question** clearly into the microphone
4. **Watch the magic happen**:
   - Your speech is transcribed in real-time
   - The system searches your knowledge base
   - You get an intelligent, summarized answer
5. **Continue the conversation** - chat history is maintained
6. **Use manual input** for text-based queries when needed

## 🔧 Configuration

### Cortex Search Setup

Your Snowflake environment needs:
```sql
-- Example Cortex Search service (adjust to your setup)
CREATE CORTEX SEARCH SERVICE haebi_cortex_search_service
ON your_documents_table
WAREHOUSE = your_warehouse;
```

### Application Configuration

All major components are configurable:

```python
# Audio processing
AudioConfig(
    chunk_duration=3,      # Seconds between transcriptions
    sample_rate=16000,     # Audio sample rate
    channels=1             # Mono audio
)

# Whisper transcription
WhisperConfig(
    model_size="base",     # tiny, base, small, medium, large
    language=None,         # Auto-detect or specify language
    fp16=True             # Use half-precision for speed
)

# Cortex Search
CortexSearchService(
    cortex_search_service="your_service_name",
    chunk_limit=2          # Number of document chunks
)
```

## 🎨 UI Features

### Enhanced Interface
- **🎤 Visual Status Indicators**: Clear feedback on what the system is doing
- **💬 Chat-Style Conversation**: Familiar messaging interface
- **📊 Real-time Transcription**: See your words as they're transcribed
- **🔍 Search Results Display**: Clear presentation of queries and answers
- **⚙️ Sidebar Controls**: Easy access to all configuration options

### Status Icons
- 🎤 **Listening**: Ready for voice input
- ⚙️ **Processing**: Transcribing speech
- 🔍 **Searching**: Querying knowledge base
- ✅ **Success**: Operation completed
- ❌ **Error**: Something went wrong

## 🚀 Advanced Features

### Multiple Input Methods
```python
# Voice input (primary)
speak_your_question()

# Manual text input (fallback)
type_your_question()

# Both methods use the same search pipeline
```

### Conversation Management
- **Chat History**: All interactions are saved in session
- **Clear Chat**: Reset conversation with one click
- **Timestamps**: Track when each interaction occurred

### Error Handling
- **Graceful Degradation**: Works with just transcription if search fails
- **Connection Testing**: Automatic Cortex Search connectivity checks
- **Fallback Options**: Multiple STUN servers for WebRTC reliability

## 🔍 Troubleshooting

### Common Issues

1. **🔴 Cortex Search: Not Connected**
   - Verify you're running in a Snowflake environment
   - Check your search service name
   - Ensure proper permissions

2. **🎤 Audio Not Working**
   - Grant microphone permissions in browser
   - Try different browsers (Chrome recommended)
   - Check WebRTC connectivity

3. **⚡ Performance Issues**
   - Use smaller Whisper models (tiny/base)
   - Adjust chunk duration
   - Check network connection

### Performance Optimization

- **🏃 Speed**: Use `tiny` or `base` Whisper models
- **🎯 Accuracy**: Use `large` Whisper model
- **⚖️ Balance**: Use `base` model (recommended)
- **🔍 Search**: Adjust chunk limit based on your needs

## 📚 Dependencies

### Core Technologies
- **Streamlit**: Web application framework
- **OpenAI Whisper**: State-of-the-art speech recognition
- **Snowflake Cortex**: Enterprise search and AI platform
- **WebRTC**: Real-time audio/video communication

### Key Libraries
- `streamlit-webrtc`: WebRTC integration for Streamlit
- `snowflake-snowpark-python`: Snowflake connectivity
- `av`: Audio/video processing
- `pydub`: Audio manipulation
- `torch`: PyTorch for ML models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

---

**🎤 Ready to revolutionize how you interact with your knowledge base? Start speaking and get instant, intelligent answers!**