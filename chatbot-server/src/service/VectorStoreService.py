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
        self.collection = "arxiv"
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

        # create client
        client = chromadb.Client(Settings(persist_directory=self.persistent_directory, anonymized_telemetry=False))
        collection = client.get_or_create_collection(self.collection)

        # Chunk text, embed, and add to vector DB
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
        embeddings = self.embeddings.encode(chunks).tolist()
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)


    def load_vector_store(self) -> Chroma:
        """
        Load the existing vector store and prepare a retriever for similarity-based lookups.

        Returns:
            Chroma: An instance of the Chroma retriever in similarity mode, ready to handle queries.
        """
        # create client
        client = chromadb.Client(Settings(persist_directory=self.persistent_directory, anonymized_telemetry=False))
        collection = client.get_or_create_collection(self.collection)

        vector_store = Chroma(
            collection_name="arxiv",
            embedding_function=self.embeddings,
            client=client,
            persist_directory=self.persistent_directory,
        )

        vector_retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.5}
        )

        return vector_retriever