import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from service.ChatbotService import ChatbotService

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    An asynchronous context manager that initializes and cleans up the ChatbotService for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded to allow the application to run within this context.

    Raises:
        Exception: Propagates any exceptions occurred during initialization or cleanup.
    """

    try:
        logger.info("Initializing Chatbot Service...")
        service = ChatbotService()
        service.initialize_service()

        # Store in app state
        app.state.service = service

        yield  # App runs here

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

app = FastAPI(lifespan=lifespan)

# enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow only frontend
    allow_credentials=True,
    allow_methods=["POST"],  # Define methods
    allow_headers=["*"],
)

@app.post("/api/chat")
async def process_chat_message(
        request: Request,
        message: str = Form(...),
        document: UploadFile = File(None)
    ) -> Dict[str, Any]:
    """
    Endpoint to process incoming chat messages and generate a response via ChatbotService.

    This function retrieves the ChatbotService instance from the application's state
    and uses it to generate an appropriate answer based on the submitted message.
    It can optionally process a PDF file as part of the summarization or question-answering.

    Args:
        message (str): The chat message text sent by the client.
        document (UploadFile, optional): An optional PDF file for summarization or Q&A.
        request (Request): The HTTP request object containing app state (e.g., ChatbotService).

    Returns:
        Dict[str, Any]: A dictionary containing the chatbot's response
                        and any relevant message content.

    Raises:
        HTTPException: If there's an error during message processing.
    """
    logger.info(f"Message: {message}")

    logger.info(f"Filename: {document.filename if document else 'No file uploaded'}")

    service: ChatbotService = request.app.state.service


    try:
        logger.info("Waiting for response...")
        content = await document.read() if document else None
        if content is not None:
            filename = document.filename if document else None
            response = json.loads(service.generate_answer(query=message, pdf_filename=filename, pdf_content=content))
        else:
            response = json.loads(service.generate_answer(query=message))
        logger.info(f"Response received: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))