import base64
import io
import logging
from PIL import Image
from typing import List, Dict, Optional
import pymupdf

logger = logging.getLogger(__name__)


class ImageUtils:
    """
    Utility class for image processing operations to support VLM functionality.
    """
    
    @staticmethod
    def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict]: List of image dictionaries with base64 data
        """
        try:
            doc = pymupdf.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = pymupdf.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Convert to base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            
                            images.append({
                                "page": page_num + 1,
                                "image_index": img_index,
                                "base64": img_base64,
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
            logger.error(f"Error extracting images from PDF: {e}")
            return []
    
    @staticmethod
    def resize_image_for_vlm(image_base64: str, max_size: int = 1024) -> str:
        """
        Resize image to optimize for VLM processing.
        
        Args:
            image_base64 (str): Base64 encoded image
            max_size (int): Maximum width or height
            
        Returns:
            str: Resized base64 encoded image
        """
        try:
            # Decode base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert back to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            resized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return resized_base64
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_base64  # Return original if resize fails
    
    @staticmethod
    def validate_image_format(image_base64: str) -> bool:
        """
        Validate if the base64 string represents a valid image.
        
        Args:
            image_base64 (str): Base64 encoded image
            
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify the image
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_base64: str) -> Optional[Dict]:
        """
        Get information about an image.
        
        Args:
            image_base64 (str): Base64 encoded image
            
        Returns:
            Optional[Dict]: Image information or None if invalid
        """
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "size_bytes": len(image_data)
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return None
    
    @staticmethod
    def create_thumbnail(image_base64: str, thumbnail_size: int = 256) -> str:
        """
        Create a thumbnail version of the image.
        
        Args:
            image_base64 (str): Base64 encoded image
            thumbnail_size (int): Size of the thumbnail
            
        Returns:
            str: Base64 encoded thumbnail
        """
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Create thumbnail
            image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return thumbnail_base64
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return image_base64  # Return original if thumbnail creation fails 