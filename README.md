# RAG Arxiv Summarization

This repository provides a chatbot application that uses a Python FastAPI backend and a React frontend. The system summarizes PDFs and handles user queries via a vector-based retrieval approach (RAG).

## Folder Structure

- **chatbot-server**  
  - **src**  
    - **main.py** – FastAPI entry point for receiving files and user messages  
    - **service**  
      - **ChatbotService.py** – Main logic for intent classification, document summarization, and Q&A  
      - **VectorStoreService.py** - Vector storage to retrieve semantic information from uploaded document by contextual question-answering (Q&A)
  - **data**  
    - **ChatMessage.py** – Data model for incoming chat messages  
  - **requirements.txt** – Python dependencies  

- **chatbot-app**  
  - **src**  
    - **App.js** – React component for the chat UI  
  - **package.json** – Frontend dependencies  

## Services

### ChatbotService
- **initialize_service**: Loads environment settings, configures the LLM client, and sets up any required vector stores.  
- **generate_answer**: Classifies user intent, checks relevance, optionally parses the PDF, and returns either a summary or a direct answer.  
- **query_groq_api**: Connects to Groq for LLM requests (summaries, Q&A, etc.).  
- **is_query_relevant**: Uses the LLM to check if user queries match the system’s context.  
- **classify_intent**: Identifies whether the user wants a summary or an answer to a question.  

### VectorStoreService
Handles vector storage and retrieval to enable searching across uploaded document contexts.

## How to Use
1. Upload a PDF (optional).  
2. Enter a query or instruction.  
3. Receive a summary (if “summarize”) or an answer (if “question”) powered by the LLM.

## Docker Deployment & Test
 ```bash
 docker build -t rag-arxiv-image .
 docker run -p 8080:8080 rag-arxiv-image
 docker run rag-arxiv-image python /home/src/tests/run_tests.py
 ```

## Installation and Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/username/rag-arxiv-summarization.git
   ```
2. **Install server requirements**
   ```bash
   cd chatbot-server
   pip install -r requirements.txt
   ```
3. **Install frontend dependencies**
   ```bash
   cd ../chatbot-app
   npm install
   ```

## Running the Application

1. **Run the FastAPI server**
   ```bash
   cd chatbot-server
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

2. **Run the React app**
   ```bash
   cd ../chatbot-app
   npm start
   ```
3. **Visit http://localhost:3000 to interact with the chatbot.**


## License
This project is licensed under the MIT License.




