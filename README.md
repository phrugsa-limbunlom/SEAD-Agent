<h1 align="center"> SEAD-Agent: RAG-Powered Document System for Structural Engineering and Architectural Design </h1>

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![React](https://img.shields.io/badge/react-18.2.0-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![Mistral AI](https://img.shields.io/badge/Mistral%20AI-VLM%20%7C%20Function%20Calling-orange.svg)

</div>

<br>
<img width="1850" height="856" alt="SEAD-Agent Interface" src="https://github.com/user-attachments/assets/c6a499f3-ab15-442b-bce4-3de23e4fb455" />
<br>

A comprehensive RAG-powered document system that combines a Python FastAPI backend with a React frontend to provide intelligent analysis of research papers from structural engineering and architectural design domains. Built with **Mistral AI's Vision Language Model (VLM) and function calling capabilities**, the system automatically determines the most appropriate tools and actions based on user queries, enabling seamless document processing, research discovery, and evidence-based design recommendations.

## 🎬 Demo

### Search Papers from ArXiv
https://github.com/user-attachments/assets/fc84010e-d598-4fb7-9522-45cc08643f0f

### Summarize Documents
https://github.com/user-attachments/assets/b15d9322-eeee-4f41-ad8e-e69925cae50d

### Recommend Design
https://github.com/user-attachments/assets/c58d35f6-d1bb-48e8-a83e-7267829f4846

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Services Architecture](#-services-architecture)
- [Data Models](#-data-models)
- [Usage Examples](#-usage-examples)
- [System Requirements](#-system-requirements)
- [Installation and Setup](#-installation-and-setup)
- [Docker Deployment](#-docker-deployment)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [License](#-license)
- [Future Enhancements](#-future-enhancements)

## 🚀 Key Features

The system uses **Mistral AI's function calling capabilities** to automatically determine the appropriate action based on user queries:

1. **Document Summarization**: Upload PDFs for intelligent analysis and summarization using Mistral's VLM (Vision Language Model) for document understanding
2. **Q&A from Vector Database**: Ask questions about uploaded documents using semantic search and vector-based retrieval
3. **ArXiv Research Search**: Automatically search and retrieve relevant research papers from arXiv when needed
4. **Design Recommendations**: Generate evidence-based structural and architectural design recommendations based on uploaded papers or research findings
5. **Multi-Modal Processing**: Analyze both text content and visual elements (images, charts, diagrams) in documents
6. **Intelligent Fallback**: Automatic fallback to arXiv search when document search returns insufficient results

## 📁 Project Structure

```
SEAD-Agent/
├── chatbot-server/                 # Python FastAPI Backend
│   ├── src/
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── service/               # Core business logic services
│   │   │   ├── chatbot_service.py  # Main chatbot orchestration
│   │   │   ├── vector_store.py    # Vector storage and retrieval
│   │   │   ├── arxiv_service.py   # Research paper fetching and processing
│   │   │   ├── design_recom.py    # Design recommendation generation
│   │   │   ├── function_calling.py # Function calling orchestration
│   │   │   └── docsum.py          # Document summarization service
│   │   ├── data/                  # Data models and schemas
│   │   │   ├── chat_message.py    # Chat message models
│   │   │   ├── chatbot_response.py # Response models
│   │   │   └── doc_data.py        # Document data models
│   │   ├── constants/             # Application constants
│   │   │   └── prompt_message.py  # Standardized prompt templates
│   │   └── utils/                 # Utility functions
│   │       ├── file_utils.py      # File operation utilities
│   │       └── image_utils.py     # Image processing utilities
│   ├── requirements.txt           # Python dependencies
│   └── model.yaml                 # Model configuration
├── chatbot-app/                   # React Frontend
│   ├── src/
│   │   ├── App.js                # Main React application
│   │   ├── App.css               # Application styles
│   │   └── ...                   # Other React components
│   └── package.json              # Frontend dependencies
├── docker-compose.yaml           # Docker Compose configuration
├── Dockerfile                    # Container configuration
├── README-docker.md             # Docker-specific documentation
└── LICENSE                      # MIT License
```

## 🛠️ Services Architecture

### Function Calling Architecture
The system uses **Mistral AI's function calling capabilities** to automatically determine which tools to use based on user queries. This eliminates the need for manual intent classification and provides more intelligent responses.

**How it works:**
1. User sends a query (with optional PDF upload)
2. System analyzes the query using Mistral's VLM model
3. Model automatically determines which functions to call based on the query context
4. Functions are executed in parallel when appropriate
5. Results are combined into a comprehensive response

**Available Functions:**
- `search_document(query)`: Search uploaded documents using vector similarity (PRIORITY function)
- `search_arxiv(query, max_results)`: Search arXiv for research papers (fallback when document search insufficient)
- `get_design_recommendations(design_query, domain)`: Generate evidence-based design recommendations
- `summarize_pdf_document(pdf_document, file_name, summary_type, max_chunks)`: Summarize PDF documents with VLM

### Core Services

#### ChatbotService
The main orchestration service that handles:
- **Function Calling**: Uses Mistral AI's function calling to automatically determine which tools to use
- **Document Processing**: Handles PDF uploads and text extraction using PyMuPDF
- **VLM Integration**: Connects with Mistral API for intelligent responses using VLM capabilities
- **Service Coordination**: Orchestrates interactions between different services through automated function calls

#### FunctionCallingService
Manages the function calling workflow:
- **Function Definitions**: Defines available functions for the model to call
- **Function Execution**: Executes called functions and processes results
- **Response Generation**: Generates structured responses with initial response, sources, and final analysis
- **Fallback Logic**: Automatically triggers arXiv search when document search returns insufficient results

#### VectorStoreService
Handles document embedding and retrieval for Q&A functionality:
- **Embedding Generation**: Uses HuggingFace models for text vectorization
- **Similarity Search**: Semantic search with configurable thresholds
- **Persistent Storage**: ChromaDB for vector storage and retrieval
- **Context Retrieval**: Fetches relevant document chunks for Q&A

#### ArxivService
Specialized service for research paper discovery and analysis:
- **Paper Search**: Intelligent search across arXiv database with relevance scoring
- **Content Enhancement**: Adds structural engineering and design keywords
- **Research Categories**: Supports physics, computer science, mathematics, and materials science
- **PDF Processing**: Downloads and extracts text from research papers

#### DesignRecommendationService
Generates evidence-based design recommendations:
- **Research Analysis**: Processes multiple papers to extract design insights
- **Theme Grouping**: Groups related research for coherent recommendations
- **Confidence Scoring**: Assesses recommendation reliability
- **Domain Specialization**: Tailored recommendations for structural and architectural design

#### DocumentSummarizationService
Handles PDF document analysis using VLM:
- **Multi-Modal Analysis**: Processes both text and visual elements
- **Text Extraction**: Extracts text content from PDFs using PyMuPDF
- **VLM Integration**: Uses Mistral's VLM for comprehensive document understanding
- **Summary Generation**: Creates brief or detailed summaries based on user preference

## 🔧 Data Models

### Document Management
- **DocumentMetadata**: Essential document metadata with processing status
- **ResearchPaperMetadata**: Specialized metadata for research papers from arXiv
- **DocumentSearchResult**: Search results with relevance scoring

### Design Recommendations
- **DesignRecommendation**: Evidence-based design recommendations with confidence scores and supporting research papers

### Communication
- **ChatMessage**: User message structure with optional document upload
- **ChatbotResponse**: Standardized response format with structured data

## 🎯 Usage Examples

The system automatically determines which functions to call based on your query using Mistral's function calling:

### 1. Document Summarization
```
Upload a PDF + Message: "Please summarize this structural engineering paper"
→ System automatically calls summarize_pdf_document() → Generates intelligent summary with design focus
```

### 2. Q&A from Vector Database
```
User: "What are the key findings in this paper?"
→ System automatically calls search_document() → Provides context-aware answers from uploaded documents
```

### 3. ArXiv Research Search
```
User: "Find recent papers on seismic design of bridges"
→ System automatically calls search_arxiv() → Returns relevant papers with abstracts and metadata
```

### 4. Design Recommendations
```
User: "Recommend foundation design for soft soil conditions"
→ System automatically calls get_design_recommendations() → Generates evidence-based recommendations with confidence scores
```

### 5. Multi-Function Queries
```
User: "Find papers on steel frame design and give me recommendations"
→ System automatically calls both search_arxiv() and get_design_recommendations() → Comprehensive research-backed response
```

### 6. Automatic Fallback
```
User: "What are the latest developments in concrete technology?"
→ System calls search_document() first → If insufficient results, automatically triggers search_arxiv() → Combined response
```

## 💻 System Requirements

### **Minimum Requirements:**
- **OS**: Windows 10, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Node.js**: 16.0 or higher
- **Memory**: 4GB RAM
- **Storage**: 2GB available space

### **Recommended Requirements:**
- **Memory**: 8GB RAM or higher
- **Storage**: 5GB available space
- **Network**: Stable internet connection for API calls

## 🚀 Installation and Setup

### Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **Git**
- **Mistral AI API Key** (required)

### Backend Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/username/SEAD-Agent.git
   cd SEAD-Agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install server dependencies**
   ```bash
   cd chatbot-server
   pip install -r requirements.txt
   ```
   
   **Key Dependencies:**
   - `mistralai>=1.0.0`: Mistral AI SDK for function calling and VLM capabilities
   - `fastapi[all]`: FastAPI framework with all optional dependencies
   - `uvicorn==0.34.0`: ASGI server for FastAPI
   - `langchain-huggingface==0.1.2`: HuggingFace embeddings integration
   - `langchain_chroma==0.2.2`: Vector store using ChromaDB
   - `arxiv==2.1.0`: ArXiv API client for research paper search
   - `PyMuPDF==1.26.0`: PDF processing and text extraction
   - `chromadb==0.4.24`: Vector database for embeddings

4. **Configure environment variables**
   Create a `.env` file in the `chatbot-server` directory:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

5. **Run the FastAPI server**
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. **Install frontend dependencies**
   ```bash
   cd ../chatbot-app
   npm install
   ```

2. **Start the React development server**
   ```bash
   npm start
   ```

3. **Access the application**
   Visit `http://localhost:3000` to interact with the chatbot.

## 🐳 Docker Deployment

### **Docker Compose (Recommended)**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

### **Individual Docker Containers**
```bash
# Build and run backend
cd chatbot-server
docker build -t sead-backend .
docker run -p 8000:8000 --env-file .env sead-backend

# Build and run frontend
cd ../chatbot-app
docker build -t sead-frontend .
docker run -p 3000:3000 sead-frontend
```

For detailed Docker setup instructions, see [README-docker.md](README-docker.md).

## 📋 API Endpoints

The FastAPI backend provides the following key endpoints:

### **Main Endpoints:**
- `POST /api/chat`: Main chat endpoint with form data support
  - **Form Fields:**
    - `message` (required): User query string
    - `document` (optional): PDF file upload for summarization/Q&A
  - **Response:** JSON with structured chatbot response including:
    - `initial_response`: Brief intent statement
    - `sources`: Array of research sources
    - `final_response`: Comprehensive analysis
    - `tool_calls`: List of executed functions

### **Response Format:**
```json
{
  "initial_response": "I'll research and analyze your query about structural design...",
  "sources": [
    {
      "id": "arxiv_1",
      "title": "Recent Advances in Seismic Design...",
      "url": "https://arxiv.org/abs/...",
      "type": "ARXIV PAPER",
      "authors": "Smith, J. et al.",
      "published": "2024-01-15"
    }
  ],
  "final_response": "Based on the research conducted, here are the key findings...",
  "tool_calls": [
    {
      "function_name": "search_arxiv",
      "display_name": "ArXiv Search",
      "description": "Searching for 10 research papers about: seismic design"
    }
  ]
}
```

## 🔧 Configuration

### **Model Configuration (`model.yaml`)**
```yaml
VLM: pixtral-12b-2409
EMBEDDING: "sentence-transformers/all-MiniLM-L6-v2"
```

### **Environment Variables**
```env
# Required
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional (for development)
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760  # 10MB
```

### **Function Calling Configuration**
The system automatically configures function calling with the following priorities:
1. **Document Search**: Primary function for uploaded documents
2. **ArXiv Search**: Fallback when document search insufficient
3. **Design Recommendations**: For specific design queries
4. **Document Summarization**: For PDF analysis requests

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

### **Planned Features:**
- **Multi-Modal Processing**: Enhanced VLM capabilities for processing images, diagrams, and technical drawings
- **Additional Research Databases**: Integration with PubMed, IEEE Xplore, and other academic databases
- **Advanced Function Calling**: More specialized functions for specific engineering domains
- **Real-time Collaboration**: Multi-user document analysis and collaborative recommendations
- **Enhanced Visualization**: Interactive charts, graphs, and technical diagrams
- **Offline Capabilities**: Local model deployment for secure environments
- **Advanced Caching**: Redis-based caching for improved response times

### **Performance Improvements:**
- **Async Processing**: Background job processing for large documents
- **Database Optimization**: Advanced indexing and query optimization
- **Model Optimization**: Fine-tuned models for specific engineering domains
- **Response Streaming**: Real-time response generation for better user experience
