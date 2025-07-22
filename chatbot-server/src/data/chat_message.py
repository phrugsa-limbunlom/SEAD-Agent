from pydantic import BaseModel
from typing import Optional
from fastapi import UploadFile

class ChatMessage(BaseModel):
    """
    A Pydantic model representing a chat message request.

    Attributes:
        message (str): The user's message content.
        document (Optional[UploadFile]): Optional PDF document for summarization or Q&A.
    """
    message: str
    document: Optional[UploadFile] = None