import logging
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class VectorStoreService:

    """
    A service that manages vector storage and retrieval for documents, such as PDFs or text chunks.
    It uses HuggingFace embeddings to encode text, then stores them in a Chroma-based database
    for similarity-based retrieval.
    """

    def __init__(self, embedding_model:str):
        """
        Initialize the VectorStoreService with an embedding model name.

        Args:
            embedding_model (str): The name of the HuggingFace embedding model used for encoding text.
        """
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.persistent_directory = "./chroma"
        self.collection = "papers"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

    def create_vector_store(self, file_name: str, text: str) -> None:
        """
        Create or update a vector store with the provided file name and string content.

        This method chunks the text into segments, encodes them using the HuggingFace embeddings,
        and stores them (along with the embeddings) in a Chroma database.

        Args:
            file_name (str): The name of the file or unique identifier for the content.
            text (str): The raw text to be chunked, embedded, and stored.
        """

        # create client with telemetry disabled
        client = chromadb.Client(Settings(
            persist_directory=self.persistent_directory, 
            anonymized_telemetry=False,
            is_persistent=True
        ))
        collection = client.get_or_create_collection(self.collection)

        # Chunk text and add to vector DB (embeddings handled automatically)
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)


    def load_vector_store(self):
        """
        Load the existing vector store and prepare a retriever for similarity-based lookups.

        Returns:
            Chroma: An instance of the Chroma retriever in similarity mode, ready to handle queries.
        """
        # create client with telemetry disabled
        client = chromadb.Client(Settings(
            persist_directory=self.persistent_directory, 
            anonymized_telemetry=False,
            is_persistent=True
        ))
        collection = client.get_or_create_collection(self.collection)

        vector_store = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            client=client,
            persist_directory=self.persistent_directory,
        )

        vector_retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.5}
        )

        return vector_retriever

    def has_documents(self) -> bool:
        """
        Check if there are any documents in the vector store.
        
        Returns:
            bool: True if documents exist, False otherwise
        """
        try:
            # create client with telemetry disabled
            client = chromadb.Client(Settings(
                persist_directory=self.persistent_directory, 
                anonymized_telemetry=False,
                is_persistent=True
            ))
            collection = client.get_or_create_collection(self.collection)
            
            # Get count of documents in collection
            count = collection.count()
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking if documents exist: {e}")
            return False