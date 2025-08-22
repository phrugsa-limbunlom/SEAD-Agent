import logging
from typing import Optional, List

import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from data.doc_data import DocumentChunk

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

    def __init__(self, embedding_model:str, persistent_directory : Optional[str] = None, collection_name : Optional[str] = None):
        """
        Initialize the vector store service and its dependencies.

        Args:
            embedding_model (str): Fully qualified name of the HuggingFace/SentenceTransformers model to use for generating embeddings.
            persistent_directory (Optional[str]): Filesystem path where the Chroma persistent database is stored.
            collection_name (Optional[str]): Name of the Chroma collection used to store document chunks and their embeddings. 

        Notes:
            - Embeddings are computed locally using `SentenceTransformer`.
            - A persistent `chromadb.PersistentClient` is created and a collection is loaded or created
              on first use.
        """
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.persistent_directory = persistent_directory or "./chroma"
        self.collection_name = collection_name or "papers"

      
        # Initialize the embedding model locally instead of in ChromaDB 
        
        self.embedder = SentenceTransformer(self.embedding_model)
        logger.info(f"Loaded embedding model: {self.embedding_model}")
       
        self._create_vector_store()

    def _create_vector_store(self):
        """Create or load the persistent Chroma vector store and target collection.

        This initializes a persistent Chroma client and attempts to load an existing collection. 
        If it does not exist, a new collection is created with the local embedding function.
        """

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persistent_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except Exception as e:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._custom_embedding_function,
                metadata={"description": "PDF document chunks with text and image content"},
            )
            logger.info(f"Created new collection '{self.collection_name}'")

    def _custom_embedding_function(self, texts):
        """Generate embeddings for a list of texts using the local `SentenceTransformer` model.

        Args:
            texts (List[str]): Input text strings to embed.

        Returns:
            List[List[float]]: Embeddings as a list of vectors (one per input text).
        """
        
        embeddings = self.embedder.encode(texts)
        return embeddings.tolist()
       

    def add_document_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks and their embeddings to the Chroma collection.

        Args:
            chunks (List[DocumentChunk]): Document chunks to add. Each chunk contains the text
                content and associated metadata (e.g., chunk type, page number).
        """

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                'chunk_type': chunk.chunk_type,
                'page_num': chunk.page_number,
                'source_page': chunk.page_number + 1
            }
            if chunk.metadata:
                metadata.update(chunk.metadata)
            metadatas.append(metadata)

        # Generate embeddings using local model
        embeddings = self._custom_embedding_function(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")

    def get_document(self, query: str):
        """Query the vector store for the most similar document chunks to the input query.

        Args:
            query (str): Natural language query to search against the stored document embeddings.

        Returns:
            dict: Chroma query result containing lists for keys: "documents", "metadatas",
            and "distances" (ordered from most to least similar).
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,  # top-k results
            include=["documents", "metadatas", "distances"]
        )

        distances = results['distances'][0]

        logger.info(f"Query: '{query}' - Distances: {[f'{d:.3f}' for d in distances]}")

        logger.info(f"Documents found: {len(results['documents'][0]) if results['documents'] else 0}")

        for i, doc in enumerate(results['documents'][0]):
            # doc_preview = doc[:1000] + "..." if len(doc) > 100 else doc
            logger.info(f"Document {i + 1}: {doc}")

        for i, metadata in enumerate(results['metadatas'][0]):
            logger.info(f"Metadata {i + 1}: {metadata}")

        return results

    def has_documents(self) -> bool:
        """
        Check if there are any documents in the vector store.

        Returns:
            bool: True if documents exist, False otherwise
        """
        try:

            collection = self.client.get_or_create_collection(self.collection_name)

            # Get count of documents in collection
            count = collection.count()
            return count > 0

        except Exception as e:
            logger.error(f"Error checking if documents exist: {e}")
            return False