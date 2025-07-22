import base64
import io
import logging
from typing import List, Dict, Optional, Any, Tuple

import fitz  # PyMuPDF
from PIL import Image
from utils.image_utils import ImageUtils
from PromptMessage import PromptMessage

logger = logging.getLogger(__name__)


class DocumentSummarizationService:
    """
    Service for summarizing documents using VLM (Vision Language Model) capabilities.
    Supports PDF documents with text and visual content (images, charts, diagrams).
    Can provide both brief and detailed summaries.
    """

    def __init__(self, vlm_client: Optional[Any] = None, vlm_model: Optional[str] = None):
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model or "pixtral-12b-2409"
        self.supported_extensions = ['.pdf', '.txt']
        self.image_utils = ImageUtils()

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text content from PDF bytes.
        
        Args:
            pdf_bytes: PDF document as bytes
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()

            doc.close()
            return text_content.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {e}")
            raise

    def extract_text_and_images_from_pdf_bytes(self, pdf_bytes: bytes) -> Tuple[str, List[Dict]]:
        """
        Extract both text and images from PDF bytes.
        
        Args:
            pdf_bytes: PDF document as bytes
            
        Returns:
            Tuple of (text_content, images_list)
        """
        try:
            # Extract text
            text_content = self.extract_text_from_pdf_bytes(pdf_bytes)

            # Extract images
            images = self.extract_images_from_pdf_bytes(pdf_bytes)

            return text_content, images
        except Exception as e:
            logger.error(f"Error extracting text and images from PDF bytes: {e}")
            raise

    def extract_images_from_pdf_bytes(self, pdf_bytes: bytes) -> List[Dict]:
        """
        Extract images from PDF bytes.
        
        Args:
            pdf_bytes: PDF document as bytes
            
        Returns:
            List of image dictionaries with base64 data
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))

                            # Convert to base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                            # Resize for VLM processing
                            resized_base64 = self.image_utils.resize_image_for_vlm(img_base64)

                            images.append({
                                "page": page_num + 1,
                                "image_index": img_index,
                                "base64": resized_base64,
                                "format": "png",
                                "width": pix.width,
                                "height": pix.height
                            })

                        pix = None  # Free memory

                    except Exception as e:
                        logger.warning(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                        continue

            doc.close()
            return images

        except Exception as e:
            logger.error(f"Error extracting images from PDF bytes: {e}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks for processing by VLM.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def summarize_chunk(self, chunk: str, summary_type: str = "brief") -> str:
        """
        Summarize a text chunk using VLM.
        
        Args:
            chunk: Text chunk to summarize
            summary_type: Type of summary ("brief" or "detailed")
            
        Returns:
            Summary of the chunk
        """
        if not self.vlm_client:
            raise ValueError("VLM client not initialized")

        if summary_type == "brief":
            prompt = f"""Please provide a concise summary of the following text, focusing on the key points and main ideas:

            {chunk}
            
            Summary:"""
        else:  # detailed
            prompt = f"""Please provide a comprehensive summary of the following text, including:
            - Main topics and themes
            - Key findings or conclusions
            - Important details and supporting information
            - Any technical terms or concepts mentioned
            
            {chunk}
            
            Detailed Summary:"""

        try:
            response = self.vlm_client.chat(
                model=self.vlm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return f"Error summarizing this section: {str(e)}"

    def summarize_chunk_with_images(self, text_chunk: str, images: List[Dict], summary_type: str = "brief") -> str:
        """summarize_document_from_bytes
        Summarize a text chunk with associated images using VLM.
        
        Args:
            text_chunk: Text chunk to summarize
            images: List of image dictionaries with base64 data
            summary_type: Type of summary ("brief" or "detailed")
            
        Returns:
            Summary of the chunk with visual content
        """
        if not self.vlm_client:
            raise ValueError("VLM client not initialized")

        # Prepare content for VLM
        content = []

        # Add text content
        if text_chunk.strip():
            content.append({
                "type": "text",
                "text": text_chunk
            })

        # Add image content (limit to 3 images per chunk to avoid token limits)
        for i, img in enumerate(images[:3]):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['base64']}",
                    "detail": "high"
                }
            })

        if summary_type == "brief":
            prompt = PromptMessage.BRIEF_DOCUMENT_SUMMARIZATION
        else:  # detailed
            PromptMessage.DETAILED_DOCUMENT_SUMMARIZATION

        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "user", "content": content}
            ]

            response = self.vlm_client.chat.complete(
                model=self.vlm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error summarizing chunk with images: {e}")
            return f"Error summarizing this section: {str(e)}"

    def create_document_chunks_with_images(self, text_content: str, images: List[Dict], chunk_size: int = 4000) -> List[
        Dict]:
        """
        Create chunks of text with associated images for processing.
        
        Args:
            text_content: Full text content
            images: List of all images from the document
            chunk_size: Maximum text size per chunk
            
        Returns:
            List of chunk dictionaries with text and associated images
        """
        # First, chunk the text
        text_chunks = self.chunk_text(text_content, chunk_size)

        # Associate images with text chunks based on page numbers
        chunks_with_images = []

        for chunk_index, text_chunk in enumerate(text_chunks):
            # Estimate which pages this chunk might cover
            # This is a simple heuristic - in practice, you might want more sophisticated mapping
            estimated_pages = self._estimate_chunk_pages(text_chunk, text_content, chunk_index, len(text_chunks))

            # Find images that belong to these pages
            chunk_images = [img for img in images if img["page"] in estimated_pages]

            chunks_with_images.append({
                "chunk_index": chunk_index + 1,
                "text": text_chunk,
                "images": chunk_images,
                "estimated_pages": estimated_pages
            })

        return chunks_with_images

    def _estimate_chunk_pages(self, chunk_text: str, full_text: str, chunk_index: int, total_chunks: int) -> List[int]:
        """
        Estimate which pages a text chunk might belong to.
        This is a heuristic approach - in a real implementation, you might want to track page boundaries.
        
        Args:
            chunk_text: The text chunk
            full_text: Full document text
            chunk_index: Index of the chunk
            total_chunks: Total number of chunks
            
        Returns:
            List of estimated page numbers
        """
        # Simple heuristic: assume pages are roughly evenly distributed
        # In a real implementation, you'd want to track actual page boundaries
        chunk_start = full_text.find(chunk_text)
        chunk_end = chunk_start + len(chunk_text)

        # Estimate pages based on position in document
        # This is approximate - for better accuracy, you'd need to track page boundaries
        total_length = len(full_text)
        start_ratio = chunk_start / total_length
        end_ratio = chunk_end / total_length

        # Assume document has roughly 10 pages (adjust based on your typical documents)
        estimated_start_page = max(1, int(start_ratio * 10))
        estimated_end_page = min(10, int(end_ratio * 10) + 1)

        return list(range(estimated_start_page, estimated_end_page + 1))

    def summarize_document_from_bytes(self, pdf_bytes: bytes, file_name: str = "document.pdf",
                                      summary_type: str = "brief", max_chunks: int = 5) -> Dict[str, Any]:
        """
        Summarize a PDF document from bytes using VLM with multimodal capabilities.
        
        Args:
            pdf_bytes: PDF document as bytes
            file_name: Name of the file (for reference)
            summary_type: Type of summary ("brief" or "detailed")
            max_chunks: Maximum number of chunks to process
            
        Returns:
            Dictionary containing summary information
        """
        try:
            # Extract both text and images from PDF
            text_content, images = self.extract_text_and_images_from_pdf_bytes(pdf_bytes)

            if not text_content.strip() and not images:
                return {
                    "error": "No content found in document",
                    "file_name": file_name,
                    "summary": "",
                    "chunks_processed": 0,
                    "images_found": 0
                }

            # Create chunks with associated images
            chunks_with_images = self.create_document_chunks_with_images(text_content, images)

            # Limit number of chunks to process
            if len(chunks_with_images) > max_chunks:
                chunks_with_images = chunks_with_images[:max_chunks]

            # Summarize each chunk with its associated images
            chunk_summaries = []
            for chunk_data in chunks_with_images:
                logger.info(
                    f"Summarizing chunk {chunk_data['chunk_index']}/{len(chunks_with_images)} with {len(chunk_data['images'])} images")

                chunk_summary = self.summarize_chunk_with_images(
                    chunk_data["text"],
                    chunk_data["images"],
                    summary_type
                )

                # Filter out base64 data from images for response
                filtered_images = []
                for img in chunk_data["images"]:
                    filtered_img = img.copy()
                    filtered_img["base64"] = "[IMAGE_DATA]"  # Replace base64 with placeholder
                    filtered_images.append(filtered_img)

                chunk_summaries.append({
                    "chunk_index": chunk_data["chunk_index"],
                    "summary": chunk_summary,
                    "original_length": len(chunk_data["text"]),
                    "images_count": len(chunk_data["images"]),
                    "estimated_pages": chunk_data["estimated_pages"],
                    "images": filtered_images  # Include filtered images without base64 data
                })

            # Combine chunk summaries if there are multiple chunks
            if len(chunk_summaries) == 1:
                final_summary = chunk_summaries[0]["summary"]
            else:
                combined_summaries = "\n\n".join([cs["summary"] for cs in chunk_summaries])
                final_summary = self.summarize_chunk_with_images(combined_summaries, [], "brief")

            # Filter out base64 data from the main images list
            filtered_main_images = []
            for img in images:
                filtered_img = img.copy()
                filtered_img["base64"] = "[IMAGE_DATA]"  # Replace base64 with placeholder
                filtered_main_images.append(filtered_img)

            return {
                "file_name": file_name,
                "summary": final_summary,
                "summary_type": summary_type,
                "chunks_processed": len(chunks_with_images),
                "total_text_length": len(text_content),
                "images_found": len(images),
                "chunk_summaries": chunk_summaries,
                "images": filtered_main_images,  # Include filtered images without base64 data
                "processing_mode": "multimodal_vlm"
            }

        except Exception as e:
            logger.error(f"Error summarizing document from bytes: {e}")
            return {
                "error": str(e),
                "file_name": file_name,
                "summary": "",
                "chunks_processed": 0,
                "images_found": 0
            }
