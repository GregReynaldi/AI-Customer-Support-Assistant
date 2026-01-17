# AI Customer Support Assistant

![Project Banner](https://neeko-copilot.bytedance.net/api/text2image?prompt=professional%20customer%20support%20AI%20assistant%20interface%20with%20chat%20and%20voice%20recording%20features%2C%20modern%20UI%2C%20blue%20color%20scheme&image_size=landscape_16_9)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The **AI Customer Support Assistant** is an advanced conversational AI application built with FastAPI that leverages Retrieval-Augmented Generation (RAG) to provide context-aware, professional responses to user queries. It also includes voice recording capabilities using OpenAI's Whisper model for speech-to-text conversion, and classification models to categorize queries by priority level, support queue, and request type.

This application is designed to simulate a professional customer support system, complete with a modern, user-friendly interface that includes a chat window, voice recording functionality, and real-time classification results.

## Key Features

### 🤖 Intelligent RAG System
- Uses Retrieval-Augmented Generation (RAG) with similarity search and relevance scoring
- Employs Ollama's Llama 3 model for generating professional, context-aware responses
- Maintains a vector database of relevant information for accurate query handling
- Implements threshold filtering to ensure only relevant documents are used

### 🎤 Voice Recording & Speech-to-Text
- Records audio from the user's microphone
- Converts speech to text using OpenAI's Whisper Large V3 model
- Automatically transcribes audio and populates the chat input field
- Supports real-time recording status and timer

### 📊 Intelligent Classification
- Categorizes queries into priority levels: High, Medium, Low
- Routes queries to appropriate support queues: Customer Service, IT Support, Product Support, Technical Support, Others
- Identifies request types: Change, Incident, Problem, Request
- Provides real-time classification results in the user interface

### 💾 Data Logging
- Automatically logs all queries, timestamps, and classification results to a CSV file
- Includes detailed information for each interaction for future analysis
- Uses human-readable labels for classification results

### 🎨 Professional User Interface
- Modern, responsive design with a clean, professional look
- Interactive chat interface with message history
- Real-time loading screen with progress indicator
- Audio wave animation during recording
- Professional email-style responses from the assistant

### 🔧 Robust Backend
- Built with FastAPI for high performance and scalability
- Uses asynchronous programming for efficient request handling
- Includes health checks and error handling
- Loads models during startup for faster runtime performance

## Tech Stack

| Category | Technology | Version/Source |
|----------|------------|----------------|
| Backend Framework | FastAPI | Python 3.10+ |
| Web Server | Uvicorn | ASGI Server |
| Frontend | HTML5, Tailwind CSS, JavaScript | - |
| Template Engine | Jinja2 | - |
| RAG & LLM | LangChain, Ollama | Llama 3:latest |
| Embeddings | Ollama Embeddings | mxbai-embed-large |
| Vector Database | ChromaDB | Local |
| Speech-to-Text | OpenAI Whisper | Large V3 (Local) |
| Classification Models | Hugging Face Transformers | Custom Trained |
| Audio Processing | Librosa | - |
| Data Handling | Pandas, NumPy | - |
| File I/O | CSV, Tempfile | - |

## Model Setup

Since the models are too large to include in the repository, you'll need to install them separately. Follow these steps to set up all required models:

### 1. Whisper Model (Speech-to-Text)

The Whisper model is used for converting speech to text. To install it:

1. **Run the audio2text.ipynb notebook**:
   ```bash
   jupyter notebook audio2text.ipynb
   ```

2. **Execute all cells in the notebook**. This will:
   - Download the Whisper Large V3 model from Hugging Face
   - Process a test audio file
   - Save the model to the `audio_model/` directory
   - Save the processor to the `audio_processor/` directory

### 2. Vector Database (RAG)

The vector database stores embedded documents for the RAG system. **This step is required because the `database_vec/` directory is not included in the repository.**

To create it:

1. **Ensure you have the dataset**:
   - Make sure `English_Dataset_Clean.csv` is present in the project root

2. **Run the vector.py script**:
   ```bash
   python vector.py
   ```

3. **This will**:
   - Load the dataset
   - Initialize Ollama embeddings (requires `mxbai-embed-large` model)
   - Create the vector database in the `database_vec/` directory
   - Split and embed documents from the dataset

### 3. Classification Models

The classification models categorize queries into levels, queues, and types. To train these models:

1. **Run the classification notebooks**:
   ```bash
   jupyter notebook LevelClassification.ipynb
   jupyter notebook QueueClassification.ipynb
   jupyter notebook TypeClassification.ipynb
   ```

2. **Execute all cells in each notebook**. This will:
   - Train the respective classification models
   - Save the models to their corresponding directories:
     - `level_classification/`
     - `queue_classification/`
     - `type_classification/`

### 4. Ollama Models

The application requires the following Ollama models:

1. **Install Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai/)

2. **Pull required models**:
   ```bash
   ollama pull llama3:latest
   ollama pull mxbai-embed-large
   ```

### 5. Verify Model Installation

After completing all steps, verify that the following directories exist:

- `audio_model/`
- `audio_processor/`
- `database_vec/`
- `level_classification/`
- `queue_classification/`
- `type_classification/`

If any directory is missing, revisit the corresponding installation step.

## Prerequisites

Before installing and running the application, ensure you have the following:

1. **Python 3.10+** installed on your system
2. **Ollama** installed and running (required for LLM and embeddings)
3. **Jupyter Notebook** installed (required for running model setup notebooks)
4. **Microphone access** for voice recording functionality
5. **Sufficient disk space** for model files (approximately 10GB)
6. **Required Python packages** (see Installation section)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/GregReynaldi/AI-Customer-Support-Assistant.git
cd AI-Customer-Support-Assistant
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install the dependencies manually:

```bash
pip install fastapi uvicorn jinja2 python-multipart torch librosa transformers langchain langchain-chroma langchain-ollama numpy pandas jupyter
```

### Step 4: Set Up Models

Follow the instructions in the [Model Setup](#model-setup) section to install all required models. This involves:

1. Running the `audio2text.ipynb` notebook to set up the Whisper model
2. **Running the `vector.py` script to create the vector database** (required - database_vec is not included in the repository)
3. Running the classification notebooks to train the classification models
4. Installing and pulling the required Ollama models

### Step 5: Verify Installation

Ensure all model directories are present and the dependencies are installed correctly:

```bash
# Check model directories
ls -la audio_model/ audio_processor/ database_vec/ level_classification/ queue_classification/ type_classification/

# Check dependencies
pip list | grep -E "fastapi|uvicorn|transformers|langchain|ollama|pandas|jupyter"
```

## Usage

### Step 1: Start the FastAPI Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:8000`. You'll see logs indicating that models are being loaded. This may take a few minutes as all models are initialized during startup.

### Step 2: Access the Application

Open your web browser and navigate to `http://localhost:8000`. You'll see a loading screen while the models are being initialized. Once loaded, the main application interface will appear.

### Step 3: Interact with the Assistant

#### Option 1: Text Chat
1. Type your question in the chat input field
2. Click "Send" or press Enter
3. Wait for the assistant to generate a response
4. View classification results in the right panel

#### Option 2: Voice Recording
1. Click the red microphone button to start recording
2. Speak your question clearly
3. Click the button again to stop recording
4. Wait for the audio to be transcribed
5. The transcription will appear in the text area
6. Click "Use in Chat" to send the transcribed text

### Step 4: View Logs

All interactions are logged to `customer_email.csv` in the project root directory. This file includes:
- Timestamp of the interaction
- User query
- Priority level classification
- Support queue classification
- Request type classification

## Project Structure

```
AI-Customer-Support-Assistant/
├── app.py                # Main FastAPI application
├── templates/            # Frontend templates
│   └── index.html        # Main UI interface
├── static/               # Static files directory
├── audio_model/          # Whisper model files
├── audio_processor/      # Whisper processor files
├── database_vec/         # ChromaDB vector database
├── level_classification/ # Level classification model
├── queue_classification/ # Queue classification model
├── type_classification/  # Type classification model
├── customer_email.csv    # Interaction log file
├── requirements.txt      # Python dependencies
└── README.md             # This README file
```

### Key Files and Directories

- **app.py**: The main FastAPI application file containing all backend logic, including model loading, API endpoints, and business logic.
- **templates/index.html**: The frontend interface built with HTML, Tailwind CSS, and JavaScript, providing the user with a professional chat and voice recording interface.
- **audio_model/**: Contains the Whisper Large V3 model files for speech-to-text conversion.
- **database_vec/**: Contains the ChromaDB vector database with embedded documents for RAG.
- **level_classification/**, **queue_classification/**, **type_classification/**: Contains the trained classification models for categorizing queries.
- **customer_email.csv**: Logs all user interactions, including timestamps and classification results.

## API Endpoints

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/` | GET | Serves the main application interface | N/A | HTML page |
| `/health` | GET | Health check endpoint | N/A | `{"status": "healthy", "models_loaded": true}` |
| `/chat` | POST | Processes text queries using RAG | `query: string` | `{"response": "Assistant response", "classifications": {...}}` |
| `/audio` | POST | Processes audio files using Whisper | `file: audio/wav` | `{"transcription": "Text transcription", "classifications": {...}}` |

## How It Works

### 1. Model Initialization
When the application starts, it loads all required models:
- **RAG Components**: ChromaDB vector database and Ollama embeddings
- **LLM**: Ollama's Llama 3 model for generating responses
- **Speech-to-Text**: Whisper Large V3 model for audio transcription
- **Classification Models**: Trained models for level, queue, and type classification

### 2. RAG Process
1. **Query Processing**: The user's query is received via the `/chat` endpoint
2. **Similarity Search**: The query is used to search the vector database for relevant documents using similarity scoring
3. **Relevance Filtering**: Documents with relevance scores above a threshold (0.4) are selected
4. **Response Generation**: The selected documents are passed to the LLM along with the query to generate a context-aware response
5. **Classification**: The query is classified into level, queue, and type categories
6. **Logging**: The interaction is logged to the CSV file
7. **Response Return**: The generated response and classification results are returned to the frontend

### 3. Voice Recording Process
1. **Audio Capture**: The user records audio via the browser's microphone
2. **Audio Processing**: The audio is captured as a blob and sent to the `/audio` endpoint
3. **Transcription**: Whisper model converts the audio to text
4. **Classification**: The transcribed text is classified
5. **Response**: The transcription and classification results are returned to the frontend
6. **User Action**: The user can then send the transcribed text as a chat query

### 4. Classification Process
1. **Text Input**: Either the user's text query or transcribed audio is received
2. **Tokenization**: The text is tokenized using the respective classification tokenizers
3. **Model Inference**: Each classification model processes the tokenized text
4. **Result Mapping**: The model outputs (numbers) are mapped to human-readable labels
5. **Result Display**: The labels are displayed in the frontend and logged to the CSV file

## Directory Structure

### Model Directories

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `audio_model/` | Whisper model for speech-to-text | Model weights and configuration files |
| `audio_processor/` | Whisper processor for audio processing | Tokenizer and feature extractor files |
| `database_vec/` | ChromaDB vector database | Embedded documents and index files |
| `level_classification/` | Level classification model | Trained model and tokenizer files |
| `queue_classification/` | Queue classification model | Trained model and tokenizer files |
| `type_classification/` | Type classification model | Trained model and tokenizer files |

### Log File

| File | Purpose | Format |
|------|---------|--------|
| `customer_email.csv` | Interaction log | CSV with timestamp, query, level, queue, type |

## Configuration

### Environment Variables

No environment variables are required for basic operation. However, you may want to configure:

- **Ollama Models**: Ensure `llama3:latest` and `mxbai-embed-large` are available in your Ollama installation
- **Model Directories**: Ensure all model directories are present in the project root

### Customization

- **Classification Threshold**: Modify the `threshold` variable in `app.py` to adjust the relevance score threshold for RAG
- **Model Paths**: Update the model paths in `app.py` if you move the model directories
- **UI Design**: Modify `templates/index.html` to customize the frontend appearance
- **Response Format**: Adjust the prompt template in `app.py` to change the assistant's response style

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Model loading failed** | Ollama models not installed | Run `ollama pull llama3:latest` and `ollama pull mxbai-embed-large` |
| **Speech-to-text not working** | Whisper model not found | Ensure `audio_model/` and `audio_processor/` directories are present |
| **Classification returning numbers** | Label mapping issue | Check the classification label mappings in `index.html` and `app.py` |
| **Microphone access denied** | Browser permission issue | Allow microphone access in your browser settings |
| **CSV logging not working** | File permission issue | Ensure the application has write access to the project directory |
| **Slow response times** | Models not fully loaded | Wait for the loading screen to complete before sending queries |

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `RAG or LLM models not loaded` | Models failed to initialize during startup | Check server logs for model loading errors |
| `Error: Could not process your request` | Backend processing error | Check server logs for detailed error information |
| `Error: Could not access microphone` | Browser microphone permission denied | Allow microphone access in browser settings |
| `Error: Could not process audio` | Audio processing error | Ensure the audio file is in a supported format |

## Contributing

Contributions are welcome! To contribute to this project:

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a new branch** for your feature or bug fix
4. **Make your changes** with clear, descriptive commits
5. **Push your changes** to your fork
6. **Submit a pull request** to the main repository

### Contribution Guidelines

- Follow the existing code style and structure
- Add comments to explain complex code
- Test your changes thoroughly
- Update the README if you modify functionality
- Ensure all dependencies are properly documented

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for the Whisper speech-to-text model
- **Ollama** for providing the Llama 3 LLM and embeddings
- **LangChain** for the RAG framework and tools
- **Hugging Face** for the Transformers library and model infrastructure
- **FastAPI** for the high-performance backend framework
- **Tailwind CSS** for the responsive frontend design
- **ChromaDB** for the vector database implementation

## Contact

For questions, feedback, or collaboration opportunities:

- **GitHub**: [GregReynaldi](https://github.com/GregReynaldi)
- **Email**: [gregoriusreynaldi@gmail.com](mailto:gregoriusreynaldi@gmail.com)

---

Thank you for using the AI Customer Support Assistant! We hope it serves your needs effectively and demonstrates the power of modern AI technologies for customer support applications.
