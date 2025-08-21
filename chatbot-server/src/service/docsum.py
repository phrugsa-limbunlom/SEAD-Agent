import os
import io
import uuid
from typing import List

import fitz
import torch
from PIL import Image
from constants.prompt_message import PromptMessage
from data.doc_data import DocumentChunk
from service.vector_store import VectorStoreService
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

import base64
import io

logger = logging.getLogger(__name__)


class DocumentSummarizationService:
    def __init__(self, vlm_client, vlm_model: str, embedding_model: str, vector_store=None):
        self.client = vlm_client
        self.vlm_model = vlm_model
        self.embedding_model = embedding_model
        
        # Use provided vector store or create one if not provided (for backward compatibility)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStoreService(embedding_model=self.embedding_model)

    def summarize_document(self, pdf_content, summary_type: str, max_chunks: int):
        """
        Summarize a document from a file path using the VLM client by processing it in chunks.

        Args:
            pdf_content:
            summary_type: Type of summary ("brief" or "detailed")
            max_chunks: Maximum number of chunks to process

        Returns:
            str: The generated summary
        """

        # Split document into chunks

        chunks = self._chunk_document(pdf_content=pdf_content)

        # save chunk
        self._save_chunk(chunks)

        # Limit the number of chunks to process
        chunks = chunks[:max_chunks]

        # Summarize each chunk individually
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_summary = self._summarize_chunk(chunk)
            logger.info(f"Chunk {i} summary: {chunk_summary}")
            chunk_summaries.append(chunk_summary)

        # Combine all chunk summaries
        if len(chunk_summaries) > 1:
            combined_summary = self._create_final_summary(chunk_summaries, summary_type)
            logger.info(f"All chunk summaries: {combined_summary}")
            return combined_summary
        else:
            return chunk_summaries[0]

    def _summarize_chunk(self, chunk):
        """
        Summarize a single chunk of the document.

        Args:
            chunk: The DocumentChunk object to summarize

        Returns:
            str: Summary of the chunk
        """

        prompt = PromptMessage.DOCUMENT_SUMMARIZATION_PROMPT
        prompt = prompt.format(text=chunk.content)
        
        logger.info(f"Summarizing chunk of type: {chunk.chunk_type}, length: {len(chunk.content)} sentences")

        response = self.client.chat.complete(
            model=self.vlm_model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that specializes in analyzing and summarizing academic papers and technical documents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512
        )

        return response.choices[0].message.content.strip()

    def _create_final_summary(self, chunk_summaries, summary_type):
        """
        Create the final document summary from individual chunk summaries.

        Args:
            chunk_summaries (List[str]): Summaries generated for each processed chunk.
            summary_type (str): Controls the style of the final summary. Use "brief" for a concise
                overview; otherwise produces a detailed summary.

        Returns:
            str: Final comprehensive summary aggregated and rewritten according to the selected style.
        """
        combined_text = "\n\n".join([f"Section {i + 1}: {summary}" for i, summary in enumerate(chunk_summaries)])

        if summary_type == "brief":
            prompt = PromptMessage.BRIEF_DOCUMENT_SUMMARIZATION
        else:
            prompt = PromptMessage.DETAILED_DOCUMENT_SUMMARIZATION

        response = self.client.chat.complete(
            model=self.vlm_model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that specializes in analyzing and summarizing academic papers and technical documents."},
                {"role": "user", "content": prompt.format(content=combined_text)},
            ],
            temperature=0.3,
            max_tokens=1024
        )

        return response.choices[0].message.content.strip()

    def _save_chunk(self, chunks: List[DocumentChunk]):
        self.vector_store.add_document_chunks(chunks)

    def _chunk_document(self, pdf_content, max_length: int = 512) -> List[DocumentChunk]:
        
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            logger.info(f"Successfully opened PDF with {len(doc)} pages")
        except Exception as e:
            raise Exception(f"Failed to open PDF content. Error: {e}")
        
        chunks = []
        
        try:
            # Load pages
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                page_num = page_num + 1 # for logging

                text = page.get_text()

                logger.info(f"Page {page_num}: extracted {len(text)} characters")

                image_list = page.get_images(full=True)
                logger.info(f"Page {page_num}: found {len(image_list)} images")

                if text and isinstance(text, str) and text.strip():

                    # Extract text in the page
                    text_chunks = self._split_text(text, max_length=max_length)
                    logger.info(f"Page: {page_num}, Text chunks: {text_chunks}")
                    logger.info(f"Page {page_num}: created {len(text_chunks)} text chunks")
                    
                    for i, chunk in enumerate(text_chunks):
                        if chunk and chunk.strip():
                            chunk_id = f"text_{page_num}_{i}_{uuid.uuid4().hex[:8]}"
                            chunks.append(DocumentChunk(
                                content=chunk,
                                chunk_type='text',
                                page_number=page_num,
                                chunk_id=chunk_id,
                                metadata={'chunk_id': f'text_{page_num}_{i}'}
                            ))
                else:
                    logger.warning(f"Page {page_num}: No valid text content found")

                if image_list:
      
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)

                            if pix.n - pix.alpha < 4:  # Gray or RGB
                                img_data = pix.tobytes("png")
                                
                                # image caption
                                caption = self._generate_image_caption(img_data)

                                logger.info(f"Page {page_num}, Image {img_index + 1}, Image caption: {caption}")

                                chunk_id = f"image_{page_num}_{img_index}_{uuid.uuid4().hex[:8]}"

                                chunks.append(DocumentChunk(
                                    content=caption,
                                    chunk_type='image',
                                    page_number=page_num,
                                    chunk_id=chunk_id,
                                    metadata={
                                        'source_page': page_num + 1,
                                        'image_index': img_index,
                                        }
                                    ))

                                pix = None
                        except Exception as e:
                            logger.warning(f"Error processing image on page {page_num}: {e}")
                    
        finally:
            doc.close()
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def _split_text(self, text: str, max_length=512) -> List[str]:
        """Split text into chunks of specified maximum length"""
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)} - {text}")
            return []

        text = text.strip()
        if not text:
            return []
        
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if sentence.strip():
                # 512 sentences in one chunk
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". " # next sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _generate_image_caption(self, image):
        
        # Encode the raw image bytes to base64
        img_base64 = base64.b64encode(image).decode('utf-8')
        
        messages = [
            {"role": "system", "content": "You are a research assistant tasked with analyzing and describing figures in research papers (chart, diagram, screenshot, etc.). Provide a clear and on point caption that describes what you see in the image."},
            {"role": "user", "content": [
                {"type": "text", "text": "Please describe this image in detail if the image is related to reseach papers, otherwise skip it (eg. logo)"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}
        ]
        
        response = self.client.chat.complete(
            model=self.vlm_model,
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()