import json
import logging
import os
from typing import Optional, Any
import pymupdf
import requests.exceptions
from constants.PromptMessage import PromptMessage
from dotenv import load_dotenv, find_dotenv
from fastapi import UploadFile, File
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from service.VectorStoreService import VectorStoreService
from utils.file_utils import FileUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    A service class that handles chatbot functionality including LLM interactions,
    relevance checking, summarization, and Q&A mechanisms.

    This service integrates with Groq API for LLM capabilities and supports
    vector search for retrieving relevant information from PDFs or queries.

    Attributes:
        template (ChatPromptTemplate): The prompt template structuring conversation flow.
        client (Groq): An API client instance for LLM interactions (Groq API).
        llm_model (str): The language model identifier used for LLM requests.
        embedding_model (str): The embedding model identifier for vector storage.
        vector (VectorStoreService): Handles vector creation, updating, and retrieval for documents.
    """

    def __init__(self, template: Optional[str] = None,
                 client: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 vector: Optional[VectorStoreService] = None):

        """
        Initialize a new ChatbotService instance.

        Args:
            template: A prompt template (if any) for structuring conversation or summarization.
            client: API client for LLM interactions (e.g., Groq).
            llm_model: The identifier for the language model used by the API.
            embedding_model: The identifier for the embedding model used in the VectorStoreService.
        """

        self.template = template
        self.client = client
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vector = vector

    def query_groq_api(self, client: Any, prompt: str, model: str) -> str:
        """
        Query the Groq API directly and return the LLM-generated text response.

        Args:
            client: The Groq API client to communicate with.
            prompt: User or system prompt text to send to the LLM.
            model: The model identifier used for text generation.

        Returns:
            str: The text response from the LLM.

        Raises:
            ValueError: If the provided prompt is not a string.
            requests.exceptions.HTTPError: If there's an HTTP error from the LLM service.
            Exception: For any other unexpected issues during the request.
        """
        try:
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt must be a string, but got {type(prompt)}")

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.5,
                max_tokens=1024,
                stop=None,
                stream=False,
            )

            return response.choices[0].message.content

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error occurred: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def is_query_relevant(self, query: str) -> bool:
        """
        Evaluate whether a user query is relevant to the system's context.

        This method forms a relevance prompt and queries the LLM to determine if
        the user's query falls within the service's defined scope.

        Args:
            query (str): The text input from the user.

        Returns:
            bool: True if the LLM deems the query relevant, otherwise False.
        """
        relevance_prompt = ChatPromptTemplate([PromptMessage.RELEVANCE_PROMPT]).invoke(
            {"template": self.template, "query": query}).to_string()

        relevance_response = self.query_groq_api(client=self.client, prompt=relevance_prompt, model=self.llm_model)

        return relevance_response == "relevant"

    def classify_intent(self, query: str) -> str:
        """
        Determine the user's intent, such as 'summarize' or 'question'.

        Uses a dedicated LLM prompt to classify the nature of the user query.

        Args:
            query (str): The text input from the user.

        Returns:
            str: The identified intent (e.g., 'summarize', 'question').
        """
        intent_prompt = ChatPromptTemplate([PromptMessage.INTENT_PROMPT]).invoke({"query": query}).to_string()

        intent = self.query_groq_api(client=self.client, prompt=intent_prompt, model=self.llm_model)

        return intent

    def generate_answer(self, query: str, pdf: UploadFile = File(None)) -> str:
        """
        Generate an answer to a user query by checking relevance, intent, and interacting with the LLM.

        Behavior:
         - If `is_query_relevant` is False, returns a default fallback message.
         - If intent is 'summarize', optionally parses a PDF to extract text.
           Creates a summary prompt and stores the PDF's content in the vector store.
         - If intent is 'question', retrieves relevant context from the vector store,
           then constructs a Q&A prompt for the LLM.
         - Otherwise returns a fallback message.

        Args:
            query (str): The user's query string.
            pdf (UploadFile, optional): An optional PDF file for summarization tasks.

        Returns:
            str: A JSON-formatted message containing the chatbot's response.
        """
        if not self.is_query_relevant(query):
            return json.dumps({"message": PromptMessage.DEFAULT_MESSAGE})

        intent = self.classify_intent(query)

        if intent == "summarize":
            text = query
            if pdf is not None:
                doc = pymupdf.open(stream=pdf.read(), filetype="pdf")
                text = "\n".join([p.get_text() for p in doc])

            summary_prompt = (ChatPromptTemplate.from_messages(
                [PromptMessage.SYSTEM_MESSAGE, PromptMessage.HUMAN_MESSAGE, PromptMessage.AI_MESSAGE]).invoke(
                {"query": text})
                              .to_string())

            summary = self.query_groq_api(client=self.client, prompt=summary_prompt, model=self.llm_model)

            self.vector.create_vector_store(pdf.filename, text)

            return json.dumps({"message": summary})

        elif intent == "question":
            retrival = self.vector.load_vector_store().invoke(query)

            context = " ".join([doc.page_content for doc in retrival])

            prompt = (ChatPromptTemplate.from_messages(
                [PromptMessage.SYSTEM_MESSAGE, PromptMessage.HUMAN_MESSAGE, PromptMessage.AI_MESSAGE]).invoke(
                {"context": context, "query": query})
                      .to_string())

            answer = self.query_groq_api(client=self.client, prompt=prompt, model=self.llm_model)

            return json.dumps({"message": answer})

        return json.dumps({"message": PromptMessage.FALL_BACK_MESSAGE})

    def initialize_service(self) -> None:
        """
        Prepare the ChatbotService for use by loading environment settings and configuring
        the LLM, embeddings, and other dependencies.

        Steps:
         - Load environment variables from .env using load_dotenv.
         - Build a default conversation prompt template from system/human/AI messages.
         - Instantiate the Groq client with the provided API key.
         - Read the model configuration from model.yaml to set LLM and embedding model fields.
         - Prepare or create any required vector store connections.

        Raises:
            FileNotFoundError: If model.yaml is missing or unreadable.
            KeyError: If the LLM or EMBEDDING keys are not found in model.yaml.
            Exception: For any other issues that occur during initialization.
        """

        logger.info("Initialize the service")

        load_dotenv(find_dotenv())

        self.template = ChatPromptTemplate.from_messages(
            [PromptMessage.SYSTEM_MESSAGE, PromptMessage.HUMAN_MESSAGE, PromptMessage.AI_MESSAGE])

        # groq API client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # model
        file_path = "/home/src/backend/model.yaml"
        model_list = FileUtils.load_yaml(file_path)

        self.llm_model = model_list["LLM"]
        self.embedding_model = model_list["EMBEDDING"]
        self.vector = VectorStoreService(embedding_model=self.embedding_model)
