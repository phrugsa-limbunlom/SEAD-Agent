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
from service.FunctionCallingService import FunctionCallingService
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
                 vlm_model: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 vector: Optional[VectorStoreService] = None):

        self.template = template
        self.client = client
        self.vlm_model = vlm_model or "pixtral-12b-2409"
        self.embedding_model = embedding_model or "mistral-embed"
        self.vector = vector
        
        # Add new services
        self.arxiv_service = ArxivService()
        self.design_recommendation_service = None 
        self.function_calling_service = None

    def get_function_definitions(self) -> List[Dict]:
        """
        Get function definitions from FunctionCallingService.
        """
        if self.function_calling_service:
            return self.function_calling_service.get_function_definitions()
        return []

    def query_mistral_with_function_calling(self, messages: List[Dict], model: str) -> Any:
        """
        Query Mistral API with function calling capabilities using FunctionCallingService.
        """
        try:
            if not self.function_calling_service:
                raise ValueError("Function calling service is not initialized")
                
            return self.function_calling_service.query_vlm_with_function_calling(messages, model)
            
        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            raise

    def process_function_calls(self, messages: List[Dict], response: Any) -> Dict:
        """
        Process function calls from model response using FunctionCallingService.
        """
        try:
            if not self.function_calling_service:
                raise ValueError("Function calling service is not initialized")
                
            return self.function_calling_service.process_function_calls(messages, response)
                
        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            return {
                "initial_response": "Processing your request...",
                "sources": [],
                "final_response": f"Error processing request: {str(e)}",
                "tool_calls": []
            }

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
                model=self.vlm_model,
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
                response = self.query_mistral_with_function_calling(messages, self.vlm_model)
                result = self.process_function_calls(messages, response)
                
                return json.dumps({"message": result["final_response"], "result": result})
            
            # For regular queries, use function calling with system prompt
            system_prompt = PromptMessage.FUNCTION_CALLING_SYSTEM_PROMPT
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.query_mistral_with_function_calling(messages, self.vlm_model)
            result = self.process_function_calls(messages, response)
            
            return json.dumps({"message": result["final_response"], "result": result})
            
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
        
        self.client = Mistral(api_key=api_key)

        # model configuration
        file_path = "/home/src/backend/model.yaml"
        if os.path.exists(file_path):
            model_config = FileUtils.load_yaml(file_path)
            if isinstance(model_config, dict):
                self.vlm_model = model_config["VLM"]
                self.embedding_model = model_config["EMBEDDING"]
    
        
        self.vector = VectorStoreService(embedding_model=self.embedding_model)
        
        # Initialize design recommendation service
        self.design_recommendation_service = DesignRecommendationService(
            arxiv_service=self.arxiv_service,
            vector_service=self.vector,
            llm_client=self.client
        )
        
        # Initialize function calling service
        self.function_calling_service = FunctionCallingService(
            arxiv_service=self.arxiv_service,
            vector_service=self.vector,
            design_recommendation_service=self.design_recommendation_service,
            vlm_client=self.client,
            vlm_model=self.vlm_model
        )
        
        logger.info(f"Enhanced chatbot service initialized with Mistral AI function calling and VLM model: {self.vlm_model}")
    

