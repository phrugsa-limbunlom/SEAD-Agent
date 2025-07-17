import json
import logging
import os
from typing import Optional, Any, List, Dict
import pymupdf
import requests.exceptions
from constants.PromptMessage import PromptMessage
from dotenv import load_dotenv, find_dotenv
from fastapi import UploadFile, File
from mistralai import Mistral
from langchain_core.prompts import ChatPromptTemplate
from service.VectorStoreService import VectorStoreService
from service.ArxivService import ArxivService
from service.DesignRecommendationService import DesignRecommendationService
from utils.file_utils import FileUtils
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    A service class that handles chatbot functionality including LLM interactions,
    function calling, summarization, and Q&A mechanisms using Mistral AI's VLM models.

    This service integrates with Mistral API for LLM capabilities and supports
    function calling for automatic tool usage based on user queries.
    """

    def __init__(self, template: Optional[str] = None,
                 client: Optional[Any] = None,
                 llm_model: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 vector: Optional[VectorStoreService] = None):

        self.template = template
        self.client = client
        self.llm_model = llm_model or "pixtral-12b-2409"  # Default to VLM model
        self.embedding_model = embedding_model
        self.vector = vector
        
        # Add new services
        self.arxiv_service = ArxivService()
        self.design_recommendation_service = None  # Will be initialized in initialize_service

    def get_function_definitions(self) -> List[Dict]:
        """
        Define the functions that the model can call automatically using Mistral's format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": "Search for research papers on arXiv. Use this when the user is looking for academic papers, research, or scientific literature on any topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding relevant research papers on arXiv"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of papers to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_document",
                    "description": "Search through uploaded documents using vector similarity. Use this when the user has questions about previously uploaded documents or PDFs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or search query about the uploaded document"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_design_recommendations",
                    "description": "Get design recommendations based on research papers and best practices. Use this when the user is asking for design advice, recommendations, or best practices in engineering or architecture.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "design_query": {
                                "type": "string",
                                "description": "The design question or area where recommendations are needed"
                            },
                            "domain": {
                                "type": "string",
                                "description": "The domain of the design (e.g., 'structural_architectural', 'mechanical', 'civil')",
                                "default": "structural_architectural"
                            }
                        },
                        "required": ["design_query"]
                    }
                }
            }
        ]

    def _execute_function(self, function_name: str, function_args: Dict) -> str:
        """
        Execute the called function with the provided arguments.
        """
        try:
            if function_name == "search_arxiv":
                return self._search_arxiv_function(
                    query=function_args.get("query", ""),
                    max_results=function_args.get("max_results", 10)
                )
            elif function_name == "search_document":
                return self._search_document_function(
                    query=function_args.get("query", "")
                )
            elif function_name == "get_design_recommendations":
                return self._get_design_recommendations_function(
                    design_query=function_args.get("design_query", ""),
                    domain=function_args.get("domain", "structural_architectural")
                )
            else:
                return f"Unknown function: {function_name}"
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return f"Error executing function: {str(e)}"

    def _search_arxiv_function(self, query: str, max_results: int = 10) -> str:
        """
        Function to search arXiv papers that the model can call.
        """
        try:
            papers = self.arxiv_service.search_papers(query, max_results=max_results)
            
            if not papers:
                return "No relevant research papers found for your query."
            
            # Format results for the model
            results = []
            for paper in papers[:max_results]:
                results.append({
                    "title": paper['title'],
                    "authors": paper['authors'],
                    "abstract": paper['abstract'][:300] + "...",
                    "arxiv_id": paper['arxiv_id'],
                    "published": paper['published'].strftime("%Y-%m-%d"),
                    "pdf_url": paper['pdf_url']
                })
            
            return json.dumps({"papers": results, "count": len(results)})
            
        except Exception as e:
            logger.error(f"Error in arXiv search function: {e}")
            return f"Error searching arXiv: {str(e)}"

    def _search_document_function(self, query: str) -> str:
        """
        Function to search uploaded documents that the model can call.
        """
        try:
            if not self.vector:
                return "Vector store is not initialized. Please upload a document first."
            
            # Load vector store and search for relevant documents
            retriever = self.vector.load_vector_store()
            # Use invoke method for langchain retrievers
            results = retriever.invoke(query)
            
            if not results:
                return "No relevant content found in uploaded documents."
            
            # Format results for the model
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.page_content,
                    "metadata": result.metadata,
                    "source": result.metadata.get("source", "Unknown")
                })
            
            return json.dumps({"documents": formatted_results, "count": len(formatted_results)})
            
        except Exception as e:
            logger.error(f"Error in document search function: {e}")
            return f"Error searching documents: {str(e)}"

    def _get_design_recommendations_function(self, design_query: str, domain: str = "structural_architectural") -> str:
        """
        Function to get design recommendations that the model can call.
        """
        try:
            if not self.design_recommendation_service:
                return "Design recommendation service is not available."
            
            recommendations = self.design_recommendation_service.generate_recommendations(
                design_query=design_query,
                domain=domain
            )
            
            if not recommendations:
                return "No design recommendations could be generated for your query."
            
            # Format recommendations for the model
            results = []
            for rec in recommendations:
                results.append({
                    "recommendation": rec.recommendation_text,
                    "domain": rec.design_domain,
                    "application": rec.application_area,
                    "confidence": rec.confidence_score,
                    "evidence": rec.evidence_strength,
                    "complexity": rec.implementation_complexity,
                    "source_papers": rec.source_papers
                })
            
            return json.dumps({"recommendations": results, "count": len(results)})
            
        except Exception as e:
            logger.error(f"Error in design recommendations function: {e}")
            return f"Error generating recommendations: {str(e)}"

    def query_mistral_with_function_calling(self, messages: List[Dict], model: str) -> Any:
        """
        Query Mistral API with function calling capabilities.
        Implements the 4-step process described in Mistral documentation.
        """
        try:
            if not self.client:
                raise ValueError("Mistral client is not initialized")
                
            # Step 1 & 2: Send user query with tools and get model response
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                tools=self.get_function_definitions(),
                tool_choice="auto",  # Let model decide when to use tools
                parallel_tool_calls=True,  # Allow parallel function calls
                temperature=0.5,
                max_tokens=1024
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            raise

    def process_function_calls(self, messages: List[Dict], response: Any) -> str:
        """
        Process function calls from model response and generate final answer.
        Implements steps 3 & 4 of Mistral's function calling process.
        """
        try:
            # Add assistant message to conversation
            messages.append(response.choices[0].message)
            
            # Step 3: Execute function calls if any
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Executing function: {function_name} with args: {function_args}")
                    
                    # Execute the function
                    function_result = self._execute_function(function_name, function_args)
                    
                    # Add function result to messages
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": function_result,
                        "tool_call_id": tool_call.id
                    })
                
                # Step 4: Generate final answer with function results
                final_response = self.client.chat.complete(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1024
                )
                
                return final_response.choices[0].message.content
            
            else:
                # No function calls, return direct response
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            return f"Error processing request: {str(e)}"

    def is_query_relevant(self, query: str) -> bool:
        """
        Evaluate whether a user query is relevant to the system's context.
        """
        try:
            if not self.client:
                return True  # Default to relevant if client not initialized
                
            relevance_prompt = ChatPromptTemplate([PromptMessage.RELEVANCE_PROMPT]).invoke(
                {"template": self.template, "query": query}).to_string()

            messages = [{"role": "user", "content": relevance_prompt}]
            response = self.client.chat.complete(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=10
            )

            return response.choices[0].message.content.strip().lower() == "relevant"
        except Exception as e:
            logger.error(f"Error checking query relevance: {e}")
            return True  # Default to relevant if check fails

    def generate_answer(self, query: str, pdf_filename: Optional[str] = None, pdf_content: Optional[bytes] = None) -> str:
        """
        Generate an answer using Mistral's function calling capabilities.
        The model will automatically determine which functions to call based on the query.
        """
        try:
            if not self.is_query_relevant(query):
                return json.dumps({"message": PromptMessage.DEFAULT_MESSAGE})

            # Handle PDF upload for summarization
            if pdf_content:
                doc = pymupdf.open(stream=pdf_content, filetype="pdf")
                text = "\n".join([p.get_text() for p in doc])
                
                # Store in vector store
                if self.vector and pdf_filename:
                    self.vector.create_vector_store(pdf_filename, text)
                
                # Summarize the document
                summary_prompt = PromptMessage.DOCUMENT_SUMMARIZATION_PROMPT.format(text=text, query=query)
                
                messages = [{"role": "user", "content": summary_prompt}]
                response = self.query_mistral_with_function_calling(messages, self.llm_model)
                final_answer = self.process_function_calls(messages, response)
                
                return json.dumps({"message": final_answer})
            
            # For regular queries, use function calling with system prompt
            system_prompt = PromptMessage.FUNCTION_CALLING_SYSTEM_PROMPT
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.query_mistral_with_function_calling(messages, self.llm_model)
            final_answer = self.process_function_calls(messages, response)
            
            return json.dumps({"message": final_answer})
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return json.dumps({"message": f"Error: {str(e)}"})

    def initialize_service(self) -> None:
        """
        Initialize the ChatbotService with Mistral AI function calling capabilities.
        """
        logger.info("Initialize the service with Mistral AI function calling support")

        load_dotenv(find_dotenv())

        self.template = ChatPromptTemplate.from_messages(
            [PromptMessage.SYSTEM_MESSAGE, PromptMessage.HUMAN_MESSAGE, PromptMessage.AI_MESSAGE])

        # Mistral API client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        # NOTE: Uncomment after installing mistralai package
        # self.client = Mistral(api_key=api_key)
        self.client = None  # Placeholder until mistralai is installed

        # model configuration
        file_path = "/home/src/backend/model.yaml"
        if os.path.exists(file_path):
            model_config = FileUtils.load_yaml(file_path)
            if isinstance(model_config, dict):
                self.llm_model = model_config.get("LLM", "pixtral-12b-2409")
                self.embedding_model = model_config.get("EMBEDDING", "mistral-embed")
            else:
                self.llm_model = "pixtral-12b-2409"  # VLM model
                self.embedding_model = "mistral-embed"
        else:
            # Default values if config file doesn't exist
            self.llm_model = "pixtral-12b-2409"  # VLM model
            self.embedding_model = "mistral-embed"
        
        self.vector = VectorStoreService(embedding_model=self.embedding_model)
        
        # Initialize design recommendation service
        self.design_recommendation_service = DesignRecommendationService(
            arxiv_service=self.arxiv_service,
            vector_service=self.vector,
            llm_client=self.client
        )
        
        logger.info(f"Enhanced chatbot service initialized with Mistral AI function calling and VLM model: {self.llm_model}")
    

