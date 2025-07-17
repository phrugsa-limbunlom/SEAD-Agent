# RAG-Powered Document System for Structural Engineering and Architectural Design

A comprehensive RAG-powered document system that combines a Python FastAPI backend with a React frontend to provide intelligent analysis of research papers from structural engineering and architectural design domains.

## 🚀 Key Features

The system uses **Mistral AI's function calling capabilities** to automatically determine the appropriate action based on user queries:

1. **Document Summarization**: Upload PDFs for intelligent analysis and summarization using Mistral's VLM (Vision Language Model) for document understanding
2. **Q&A from Vector Database**: Ask questions about uploaded documents using semantic search and vector-based retrieval
3. **ArXiv Research Search**: Automatically search and retrieve relevant research papers from arXiv when needed
4. **Design Recommendations**: Generate evidence-based structural and architectural design recommendations based on uploaded papers or research findings

## 📁 Project Structure

```
structural-rag-agent/
├── chatbot-server/                 # Python FastAPI Backend
│   ├── src/
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── service/               # Core business logic services
│   │   │   ├── ChatbotService.py  # Main chatbot orchestration
│   │   │   ├── VectorStoreService.py  # Vector storage and retrieval
│   │   │   ├── ArxivService.py    # Research paper fetching and processing
│   │   │   └── DesignRecommendationService.py  # Design recommendation generation
│   │   ├── data/                  # Data models and schemas
│   │   │   ├── ChatMessage.py     # Chat message models
│   │   │   ├── ChatbotResponse.py # Response models
│   │   │   └── DocumentMetadata.py # Document metadata management
│   │   ├── constants/             # Application constants
│   │   │   └── PromptMessage.py   # Standardized prompt templates
│   │   └── utils/                 # Utility functions
│   │       └── file_utils.py      # File operation utilities
│   ├── requirements.txt           # Python dependencies
│   └── model.yaml                 # Model configuration
├── chatbot-app/                   # React Frontend
│   ├── src/
│   │   ├── App.js                # Main React application
│   │   └── ...                   # Other React components
│   └── package.json              # Frontend dependencies
├── tests/                        # Test files
├── Dockerfile                    # Container configuration
└── docker-entrypoint.sh         # Container startup script
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
- `search_arxiv(query, max_results)`: Search arXiv for research papers
- `search_document(query)`: Search uploaded documents using vector similarity
- `get_design_recommendations(design_query, domain)`: Generate evidence-based design recommendations

### ChatbotService
The main orchestration service that handles:
- **Function Calling**: Uses Mistral AI's function calling to automatically determine which tools to use
- **Document Processing**: Handles PDF uploads and text extraction using PyMuPDF
- **LLM Integration**: Connects with Mistral API for intelligent responses using VLM capabilities
- **Service Coordination**: Orchestrates interactions between different services through automated function calls

**Key Methods:**
- `generate_answer()`: Main entry point for processing user queries with function calling
- `query_mistral_with_function_calling()`: Interfaces with Mistral API using function calling
- `process_function_calls()`: Executes called functions and generates final responses
- `get_function_definitions()`: Defines available functions for the model to call
- `is_query_relevant()`: Validates query relevance to structural/architectural design

**Available Functions:**
- `search_arxiv()`: Search for research papers on arXiv
- `search_document()`: Search through uploaded documents using vector similarity
- `get_design_recommendations()`: Get design recommendations based on research

### VectorStoreService
Handles document embedding and retrieval for Q&A functionality:
- **Embedding Generation**: Uses HuggingFace models for text vectorization
- **Similarity Search**: Semantic search with configurable thresholds
- **Persistent Storage**: ChromaDB for vector storage and retrieval
- **Context Retrieval**: Fetches relevant document chunks for Q&A

### ArxivService
Specialized service for research paper discovery and analysis:
- **Paper Search**: Intelligent search across arXiv database with relevance scoring
- **Content Enhancement**: Adds structural engineering and design keywords
- **Research Categories**: Supports physics, computer science, mathematics, and materials science
- **PDF Processing**: Downloads and extracts text from research papers

**Key Features:**
- Supports research categories relevant to structural engineering and architecture
- Relevance scoring based on title, abstract, and category matching
- Full-text extraction from research papers

### DesignRecommendationService
Generates evidence-based design recommendations:
- **Research Analysis**: Processes multiple papers to extract design insights
- **Theme Grouping**: Groups related research for coherent recommendations
- **Confidence Scoring**: Assesses recommendation reliability
- **Domain Specialization**: Tailored recommendations for structural and architectural design

**Recommendation Domains:**
- **Structural**: Beam design, foundations, load-bearing systems
- **Architectural**: Space planning, materials, environmental considerations

## 🔧 Data Models

### Document Management
- **DocumentMetadata**: Essential document metadata with processing status
- **ResearchPaperMetadata**: Specialized metadata for research papers from arXiv
- **DocumentSearchResult**: Search results with relevance scoring

### Design Recommendations
- **DesignRecommendation**: Evidence-based design recommendations with confidence scores and supporting research papers

### Communication
- **ChatMessage**: User message structure
- **ChatbotResponse**: Standardized response format

## 🎯 Usage Examples

The system automatically determines which functions to call based on your query using Mistral's function calling:

### 1. Document Summarization
```
Upload a PDF + Message: "Please summarize this structural engineering paper"
→ System automatically analyzes content using VLM → Generates intelligent summary with design focus
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

## 🐳 Docker Deployment

Build and run the application using Docker:

```bash
# Build the Docker image
docker build -t rag-arxiv-image .

# Run the application
docker run -p 8080:8080 rag-arxiv-image

# Run tests
docker run rag-arxiv-image python /home/src/tests/run_tests.py
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/username/rag-arxiv-summarization.git
   cd rag-arxiv-summarization
   ```

2. **Install server dependencies**
   ```bash
   cd chatbot-server
   pip install -r requirements.txt
   ```
   
   **Key Dependencies:**
   - `mistralai>=1.0.0`: Mistral AI SDK for function calling and VLM capabilities
   - `groq==0.18.0`: Groq SDK (legacy support)
   - `langchain-huggingface==0.1.2`: HuggingFace embeddings integration
   - `langchain_chroma==0.2.2`: Vector store using ChromaDB
   - `arxiv==2.1.0`: ArXiv API client for research paper search
   - `PyMuPDF==1.26.0`: PDF processing and text extraction

3. **Configure environment variables**
   Create a `.env` file in the `chatbot-server` directory:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

4. **Run the FastAPI server**
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

## 📋 API Endpoints

The FastAPI backend provides the following key endpoints:

- `POST /api/chat`: Main chat endpoint with form data support
  - **Form Fields:**
    - `message` (required): User query string
    - `document` (optional): PDF file upload for summarization/Q&A
  - **Response:** JSON with chatbot response using function calling
- `GET /`: Serves the React frontend
- Static file serving for the frontend application

## 🧪 Testing

Run the test suite:
```bash
cd tests
python run_tests.py
```

## 🔧 Configuration

### Model Configuration (`model.yaml`)
```yaml
LLM: "llama-3.3-70b-versatile"
EMBEDDING: "sentence-transformers/all-MiniLM-L6-v2"
```

### Environment Variables
- `MISTRAL_API_KEY`: API key for Mistral AI service (required for function calling and VLM capabilities)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- **Multi-Modal Processing**: Enhanced VLM capabilities for processing images, diagrams, and technical drawings
- **Additional Research Databases**: Integration with PubMed, IEEE Xplore, and other academic databases
- **Advanced Function Calling**: More specialized functions for specific engineering domains
- **Real-time Collaboration**: Multi-user document analysis and collaborative recommendations
- **Enhanced Visualization**: Interactive charts, graphs, and technical diagrams
- **Offline Capabilities**: Local model deployment for secure environments




