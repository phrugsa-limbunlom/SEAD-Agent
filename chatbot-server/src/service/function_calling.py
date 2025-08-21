import json
import logging
import base64
import tempfile
from typing import Optional, Any, List, Dict, Tuple
from datetime import datetime
from service.vector_store import VectorStoreService
from service.arxiv_service import ArxivService
from service.design_recom import DesignRecommendationService
from service.docsum import DocumentSummarizationService
from constants.prompt_message import PromptMessage
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class FunctionCallingService:
    """
    A service class that handles function calling capabilities for the chatbot.
    This service manages function definitions, execution, and processing of function calls.
    Enhanced to return structured responses in PickSmart format.
    """

    def __init__(self,
                 arxiv_service: Optional[ArxivService] = None,
                 vector_service: Optional[VectorStoreService] = None,
                 design_recommendation_service: Optional[DesignRecommendationService] = None,
                 document_summarization_service: Optional[DocumentSummarizationService] = None,
                 vlm_client: Optional[Any] = None,
                 vlm_model: Optional[str] = None):

        self.arxiv_service = arxiv_service
        self.vector_service = vector_service
        self.design_recommendation_service = design_recommendation_service
        self.document_summarization_service = document_summarization_service
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model or "pixtral-12b-2409"

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert objects with datetime fields to JSON-serializable format
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")
        elif hasattr(obj, '__dict__'):
            # Handle Pydantic models or custom objects
            if hasattr(obj, 'dict'):
                # Pydantic model
                data = obj.dict()
                return self._make_serializable(data)
            else:
                # Regular object with __dict__
                data = obj.__dict__
                return self._make_serializable(data)
        else:
            return obj

    def get_function_definitions(self) -> List[Dict]:
        """
        Return function definitions for Mistral function calling.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": "Search for research papers on arXiv related to structural engineering, architecture, or related topics. This function returns academic papers with abstracts, authors, and publication details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for academic papers (e.g., 'seismic design', 'structural optimization', 'architectural analysis')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of papers to return (default: 10, max: 20)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_document",
                    "description": "Search through uploaded documents using semantic search. Useful for finding specific information within previously uploaded research papers or documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant content in uploaded documents"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_design_recommendations",
                    "description": "Generate evidence-based design recommendations for structural engineering and architectural projects based on research findings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "design_query": {
                                "type": "string",
                                "description": "Specific design question or requirement (e.g., 'foundation design for soft soil', 'seismic retrofitting strategies')"
                            },
                            "domain": {
                                "type": "string",
                                "description": "Design domain",
                                "enum": ["structural_engineering", "architectural_design", "structural_architectural"]
                            }
                        },
                        "required": ["design_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_pdf_document",
                    "description": "Summarize a PDF document provided as base64-encoded bytes using VLM with multimodal capabilities. Analyzes both text content and visual elements (images, charts, diagrams) to provide comprehensive summaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pdf_document": {
                                "type": "string",
                                "description": "PDF document as base64-encoded string"
                            },
                            "file_name": {
                                "type": "string",
                                "description": "Name of the PDF file (for reference)",
                                "default": "document.pdf"
                            },
                            "summary_type": {
                                "type": "string",
                                "description": "Type of summary to generate",
                                "enum": ["brief", "detailed"],
                                "default": "brief"
                            },
                            "max_chunks": {
                                "type": "integer",
                                "description": "Maximum number of text chunks to process (default: 5, max: 10)",
                                "default": 5
                            }
                        },
                        "required": ["pdf_document"]
                    }
                }
            }
        ]

    def execute_function(self, function_name: str, function_args: Dict) -> str:
        """
        Execute the called function with the provided arguments.
        """
        try:
            if function_name == "search_arxiv":
                return self._search_arxiv_function(
                    query=function_args.get("query", ""),
                    max_results=function_args.get("max_results", 10)
                )
            elif function_name == "search_document":
                return self._search_document_function(
                    query=function_args.get("query", "")
                )
            elif function_name == "get_design_recommendations":
                return self._get_design_recommendations_function(
                    design_query=function_args.get("design_query", ""),
                    domain=function_args.get("domain", "structural_architectural")
                )
            elif function_name == "summarize_pdf_document":
                return self._summarize_document_function(
                    pdf_document=function_args.get("pdf_document", ""),
                    file_name=function_args.get("file_name", "document.pdf"),
                    summary_type=function_args.get("summary_type", "brief"),
                    max_chunks=function_args.get("max_chunks", 5)
                )
            else:
                return f"Unknown function: {function_name}"
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return f"Error executing function: {str(e)}"

    def _search_arxiv_function(self, query: str, max_results: int = 10) -> str:
        """
        Search arXiv for research papers
        """
        try:
            if not self.arxiv_service:
                return json.dumps({"error": "ArXiv service not initialized"})

            papers = self.arxiv_service.search_papers(query, max_results)

            # Use helper method to handle datetime serialization
            result_data = {
                "papers": self._make_serializable(papers),
                "count": len(papers),
                "query": query
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return json.dumps({"error": str(e), "papers": [], "count": 0})

    def _search_document_function(self, query: str) -> str:
        """
        Search uploaded documents using vector store
        """
        try:
            result = self.vector_service.get_document(query)

            result_data = {
                "documents": self._make_serializable(result['documents'][0][0]),
                "count": len(result),
                "query": query
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return json.dumps({"error": str(e), "documents": [], "count": 0})

    def _get_design_recommendations_function(self, design_query: str, domain: str = "structural_architectural") -> str:
        """
        Get design recommendations based on research
        """
        try:
            if not self.design_recommendation_service:
                return json.dumps({"error": "Design recommendation service not initialized"})

            recommendations = self.design_recommendation_service.generate_recommendations(design_query, domain)

            result_data = {
                "recommendations": self._make_serializable(recommendations),
                "query": design_query,
                "domain": domain
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"Design recommendation error: {e}")
            return json.dumps({"error": str(e), "recommendations": [], "query": design_query})

    def _summarize_document_function(self, pdf_document, file_name, summary_type, max_chunks) -> str:
        """
        Summarize a PDF document  using the document summarization service
        """
        try:
            summary_result = self.document_summarization_service.summarize_document(
                pdf_content=pdf_document,
                summary_type=summary_type,
                max_chunks=max_chunks
            )
            result_data = {
                "summary": self._make_serializable(summary_result),
                "file_name": file_name,
                "summary_type": summary_type,
                "max_chunks": max_chunks,
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"PDF document summarization error: {e}")
            return json.dumps({"error": str(e), "summary": "", "file_name": file_name})


    def generate_initial_response(self, query: str) -> str:
        """
        Generate an initial response that shows what the AI is planning to do.
        This follows the pattern of showing analysis intent using LLM.
        """

        system_prompt = PromptMessage.INITIAL_RESPONSE_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate an initial response for this query: {query}"}
        ]

        response = self.vlm_client.chat.complete(
            model=self.vlm_model,
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    def _execute_function(self, messages: List[Dict], response: Any, pdf_data: Optional[Dict] = None,
                          tool_calls_made=None) -> Any:
        # Initialize sources list and tool calls list
        sources = []
        tool_calls_made = [] if tool_calls_made is None else tool_calls_made

        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.info(f"Executing function: {function_name} with args: {function_args}")

            # Track the tool call for UI display (filter out base64 data)
            display_args = function_args.copy()
            if function_name == "summarize_pdf_document" and "pdf_document" in display_args:
                # Replace base64 data with a placeholder for display
                display_args["pdf_document"] = "[PDF_DATA]"

            tool_call_info = {
                "function_name": function_name,
                "function_args": display_args,
                "display_name": self._get_tool_display_name(function_name),
                "description": self._get_tool_description(function_name, function_args)
            }
            tool_calls_made.append(tool_call_info)

            # Handle PDF data injection for summarize_pdf_document function
            if function_name == "summarize_pdf_document" and pdf_data:
                function_args["pdf_document"] = pdf_data["pdf_content"]
                function_args["file_name"] = pdf_data["filename"]

                logger.info(f"Injected PDF data for summarize_pdf_document function: {function_args['file_name']}")

            # Execute the function
            function_result = self.execute_function(function_name, function_args)

            # Extract sources if this is an ArXiv search
            if function_name == "search_arxiv":
                try:
                    result_data = json.loads(function_result)
                    if "papers" in result_data:
                        for i, paper in enumerate(
                                result_data["papers"][:7]):  # Limit to 7 sources
                            source_item = {
                                "id": f"arxiv_{i + 1}",
                                "title": paper["title"][:100] + "..." if len(paper["title"]) > 100 else paper[
                                    "title"],
                                "url": paper["pdf_url"],
                                "type": "ARXIV PAPER",
                                "authors": ", ".join(paper["authors"][:3]) + (
                                    " et al." if len(paper["authors"]) > 3 else ""),
                                "published": paper["published"]
                            }
                            sources.append(json.dumps(source_item))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing ArXiv results: {e}")

            # Add function result to messages
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": function_result,
                "tool_call_id": tool_call.id
            })

            return sources, tool_calls_made, messages

    def process_function_calls(self, messages: List[Dict], response: Any, pdf_data: Optional[Dict] = None) -> Dict:
        """
        Process function calls from model response and generate structured response.
        Returns structured response with initial_response, sources, and final_response.

        Args:
            messages: List of conversation messages
            response: VLM response with potential function calls
            pdf_data: Optional PDF data containing raw PDF content and filename
        """

        final_response = ""

        # Extract the original query from messages
        original_query = ""
        try:
            for msg in messages:
                if msg.get("role") == "user":
                    original_query = msg.get("content", "")
                    break

            # Generate initial response showing intent
            initial_response = self.generate_initial_response(original_query)

            # Add assistant message to conversation
            messages.append(response.choices[0].message)

            # Execute function calls if any
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:

                sources, tool_calls_made, messages = self._execute_function(messages, response, pdf_data)

                logger.info(f"Messages list before final response from Assistant: {messages}")

                response = self.vlm_client.chat.complete(
                    model=self.vlm_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1024
                )

                if hasattr(response.choices[0].message, 'tool_calls') and response.choices[
                    0].message.tool_calls is not None:
                    # call search_arxiv if search_document failed

                    logger.info("No relevant document found.")
                    logger.info("Call search_arxiv tool")

                    messages.append(response.choices[0].message)

                    logger.info(f"List of messages after final response from Assistant: {messages}")

                    sources, tool_calls_made, messages = self._execute_function(messages, response, pdf_data,
                                                                                tool_calls_made)

                # Add system prompt to messages for final response
                final_messages = [{"role": "system", "content": PromptMessage.FINAL_SYSTEM_PROMPT}] + messages

                logger.info(f"final_messages: {final_messages}")

                # Generate final answer with function results
                try:
                    final_response = self.vlm_client.chat.complete(
                        model=self.vlm_model,
                        messages=final_messages,
                        temperature=0.5,
                        max_tokens=1024
                    )
                except Exception as e:
                    logger.error(f"Error calling API for Final Response: {e}")

                logger.info(f"tool_calls_made: {tool_calls_made}")
                return {
                    "initial_response": initial_response,
                    "sources": sources,
                    "final_response": final_response.choices[0].message.content,
                    "tool_calls": tool_calls_made
                }
            else:
                # No function calls, return direct response with structure
                return {
                    "initial_response": initial_response,
                    "sources": [],
                    "final_response": response.choices[0].message.content,
                    "tool_calls": []
                }

        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            return {
                "initial_response": f"I'll analyze your query: {original_query}",
                "sources": [],
                "final_response": f"I encountered an error while processing your request. Please try again or rephrase your question.",
                "tool_calls": []
            }

    def _get_tool_display_name(self, function_name: str) -> str:
        """
        Get user-friendly display name for tool calls
        """
        display_names = {
            "search_arxiv": "ArXiv Search",
            "search_document": "Document Search",
            "get_design_recommendations": "Design Recommendations",
            "summarize_pdf_document": "Document Summarization"
        }
        return display_names.get(function_name, function_name.replace("_", " ").title())

    def _get_tool_description(self, function_name: str, function_args: Dict) -> str:
        """
        Get descriptive text for what the tool is doing
        """
        if function_name == "search_arxiv":
            query = function_args.get("query", "")
            max_results = function_args.get("max_results", 10)
            return f"Searching for {max_results} research papers about: {query}"
        elif function_name == "search_document":
            query = function_args.get("query", "")
            return f"Searching uploaded documents for: {query}"
        elif function_name == "get_design_recommendations":
            design_query = function_args.get("design_query", "")
            domain = function_args.get("domain", "structural_architectural")
            return f"Generating {domain} design recommendations for: {design_query}"
        elif function_name == "summarize_pdf_document":
            file_name = function_args.get("file_name", "document.pdf")
            summary_type = function_args.get("summary_type", "brief")
            return f"Generating {summary_type} multimodal summary of PDF document (text + images): {file_name}"
        else:
            return f"Executing {function_name}"

    def query_vlm_with_function_calling(self, messages: List[Dict], model: str) -> Any:
        """
        Query vlm model with function calling capabilities.
        """
        try:
            if not self.vlm_client:
                raise ValueError("Mistral client is not initialized")

            response = self.vlm_client.chat.complete(
                model=model,
                messages=messages,
                tools=self.get_function_definitions(),
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0.5,
                max_tokens=1024
            )

            return response

        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            raise
