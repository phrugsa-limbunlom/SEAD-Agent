from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class FunctionCallResult(BaseModel):
    """
    A Pydantic model representing the result of function calls.
    
    Attributes:
        initial_response (str): The initial response from the model.
        sources (List[str]): List of sources used.
        final_response (str): The final processed response.
        tool_calls (List[Dict[str, Any]]): List of tool calls made.
    """
    initial_response: str
    sources: List[str]
    final_response: str
    tool_calls: List[Dict[str, Any]]

class ChatbotResponse(BaseModel):
    """
    A Pydantic model representing a chatbot response.

    Attributes:
        message (str): The main response message.
        result (Optional[FunctionCallResult]): Optional detailed result for complex responses.
    """
    message: str
    result: Optional[FunctionCallResult] = None