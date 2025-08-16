import json
import logging
import os
from typing import Optional, Any, List, Dict

import pymupdf
from constants.prompt_message import PromptMessage
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from mistralai import Mistral
from service.arxiv_service import ArxivService
from service.design_recom import DesignRecommendationService
from service.docsum import DocumentSummarizationService
from service.function_calling import FunctionCallingService
from service.vector_store import VectorStoreService
from utils.file_utils import FileUtils

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
                 embedding_model: Optional[str] = None):

        self.template = template
        self.client = client
        self.vlm_model = vlm_model or "pixtral-12b-2409"
        self.embedding_model = embedding_model
        
        # Add new services
        self.arxiv_service = ArxivService()
        self.vector = None
        self.design_recommendation_service = None
        self.document_summarization_service = None
        self.function_calling_service = None

    def get_function_definitions(self) -> List[Dict]:
        """
        Get function definitions from FunctionCallingService.
        """
        if self.function_calling_service:
            return self.function_calling_service.get_function_definitions()
        return []

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
             
            # extract first page from the document to verify topic is relevant
            text = ""
            if pdf_content:
                doc = pymupdf.open(stream=pdf_content, filetype="pdf")
                if len(doc) > 0:
                    first_page = doc[0]
                    text = first_page.get_text()
                doc.close()
                
            if text is not None:
                if not self.is_query_relevant(query + text):
                    return json.dumps({"message": PromptMessage.DEFAULT_MESSAGE})
            else:
                 if not self.is_query_relevant(query):
                    return json.dumps({"message": PromptMessage.DEFAULT_MESSAGE})

            has_document = self.vector.has_documents()

            system_prompt = PromptMessage.FUNCTION_CALLING_SYSTEM_PROMPT.format(HAS_DOCUMENT=has_document)
            
            # Prepare user message with PDF context if available
            user_content = query
            if pdf_content:
                user_content = f"{query}\n\n[PDF Document Available as this file name: {pdf_filename}]"

                # Store PDF data for function calling
                self._current_pdf_data = {
                    'pdf_content': pdf_content,
                    'filename': pdf_filename
                }

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = ""
            try:
                response = self.function_calling_service.query_vlm_with_function_calling(messages, self.vlm_model)
            except Exception as e:
                logger.error(f"Error querying Mistral API: {e}")

            result = dict()
            try:
                result = self.function_calling_service.process_function_calls(messages, response, pdf_data=self._current_pdf_data if pdf_content else None)
            except Exception as e:
                logger.error(f"Error processing function calls: {e}")

            return json.dumps({"message": result["final_response"], "result": result})
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return json.dumps({"message": f"Internal Server Error. Please try again later."})

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
        file_path = "/app/model.yaml"
        if os.path.exists(file_path):
            model_config = FileUtils.load_yaml(file_path)
            if isinstance(model_config, dict):
                self.vlm_model = model_config["VLM"]
                self.embedding_model = model_config["EMBEDDING"]
    
        # Create a single vector store instance to be shared across services
        self.vector = VectorStoreService(embedding_model=self.embedding_model)

        self.arxiv_service = ArxivService()

        # Initialize design recommendation service
        self.design_recommendation_service = DesignRecommendationService(
            arxiv_service=self.arxiv_service,
            vector_service=self.vector,
            llm_client=self.client
        )

        # Initialize document summarization service with the shared vector store
        self.document_summarization_service = DocumentSummarizationService(
            vlm_client=self.client,
            vlm_model=self.vlm_model,
            embedding_model=self.embedding_model,
            vector_store=self.vector  # Pass the shared instance
        )

        # Initialize function calling service
        self.function_calling_service = FunctionCallingService(
            arxiv_service=self.arxiv_service,
            vector_service=self.vector,
            design_recommendation_service=self.design_recommendation_service,
            document_summarization_service=self.document_summarization_service,
            vlm_client=self.client,
            vlm_model=self.vlm_model
        )

        logger.info(f"Enhanced chatbot service initialized with Mistral AI function calling and VLM model: {self.vlm_model}")
    

