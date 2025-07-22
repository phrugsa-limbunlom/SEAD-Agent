import json
import logging
import base64
from typing import Optional, Any, List, Dict
from datetime import datetime
from service.vector_store import VectorStoreService
from service.arxiv_service import ArxivService
from service.design_recom import DesignRecommendationService
from service.docsum import DocumentSummarizationService
from constants.prompt_message import PromptMessage

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
                    "name": "search_document",
                    "description": "Search through uploaded documents using semantic search. This is the PRIMARY search method - use this first to find information in documents the user has already uploaded. Only use search_arxiv if this doesn't provide sufficient information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find information in uploaded documents (e.g., 'seismic design', 'structural optimization', 'architectural analysis')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": "Search for research papers on arXiv related to structural engineering, architecture, or related topics. Use this ONLY if search_document doesn't provide sufficient information. This function returns academic papers with abstracts, authors, and publication details.",
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
            if not self.vector_service:
                return json.dumps({"error": "Vector service not initialized"})

            vector_retriever = self.vector_service.load_vector_store()
            docs = vector_retriever.invoke(query)

            document_results = []
            for doc in docs:
                document_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "relevance_score": doc.metadata.get("score", 0.0)
                })

          
            result_data = {
                "documents": self._make_serializable(document_results),
                "count": len(document_results),
                "query": query,
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return json.dumps({"error": str(e), "documents": [], "count": 0, "sufficient_results": False})

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

    def _summarize_document_function(self, pdf_document: str, file_name: str, summary_type: str = "brief", max_chunks: int = 5) -> str:
        """
        Summarize a PDF document from base64-encoded bytes using the document summarization service
        """
        try:
            if not self.document_summarization_service:
                return json.dumps({"error": "Document summarization service not initialized"})

            # Validate max_chunks
            if max_chunks > 10:
                max_chunks = 10
            elif max_chunks < 1:
                max_chunks = 1

            # Decode base64 PDF document with proper padding
            try:
                # Clean the base64 string and remove data URL prefix if present
                pdf_document = pdf_document.strip()
                logger.info(f"Base64 string length: {len(pdf_document)}, first 50 chars: {pdf_document[:50]}...")
                
                # Remove data URL prefix if present (e.g., "data:application/pdf;base64,")
                if pdf_document.startswith("data:"):
                    # Extract the base64 part after the comma
                    pdf_document = pdf_document.split(",", 1)[1]
                    logger.info(f"Removed data URL prefix, new length: {len(pdf_document)}")
                
                # Add padding if needed
                padding_needed = len(pdf_document) % 4
                if padding_needed:
                    pdf_document += '=' * (4 - padding_needed)
                
                # Try to decode
                pdf_bytes = base64.b64decode(pdf_document)
            except Exception as e:
                logger.error(f"Error decoding base64 PDF document: {e}")
                # Try alternative decoding methods
                try:
                    # Try with urlsafe base64
                    pdf_bytes = base64.urlsafe_b64decode(pdf_document + '=' * (4 - len(pdf_document) % 4))
                except Exception as e2:
                    logger.error(f"Alternative base64 decoding also failed: {e2}")
                    return json.dumps({"error": "Invalid base64-encoded PDF document", "summary": "", "file_name": file_name})

            # Extract text content from PDF for vector store
            text_content = self.document_summarization_service.extract_text_from_pdf_bytes(pdf_bytes)

            summary_result = self.document_summarization_service.summarize_document_from_bytes(
                pdf_bytes=pdf_bytes,
                file_name=file_name,
                summary_type=summary_type,
                max_chunks=max_chunks
            )

            # Create vector store with the extracted text content
            if self.vector_service and text_content.strip():
                try:
                    self.vector_service.create_vector_store(file_name, text_content)
                    logger.info(f"Successfully created vector store for file: {file_name}")
                except Exception as e:
                    logger.error(f"Error creating vector store for file {file_name}: {e}")

            result_data = {
                "summary": self._make_serializable(summary_result),
                "file_name": file_name,
                "summary_type": summary_type,
                "max_chunks": max_chunks,
                "vector_store_created": self.vector_service is not None and bool(text_content.strip())
            }

            return json.dumps(result_data)
        except Exception as e:
            logger.error(f"PDF document summarization error: {e}")
            return json.dumps({"error": str(e), "summary": "", "file_name": file_name})

    def _check_and_trigger_arxiv_fallback(self, document_result: str, original_query: str) -> Optional[str]:
        """
        Check if document search returned insufficient results and trigger arXiv fallback.
        
        Args:
            document_result: JSON string result from document search
            original_query: The original user query
            
        Returns:
            arXiv search result JSON string if fallback was triggered, None otherwise
        """
        try:
            result_data = json.loads(document_result)
            document_count = result_data.get("count", 0)
            
            # If no documents found or very few results, trigger arXiv search
            if document_count == 0 or document_count < 2:
                logger.info(f"Document search returned {document_count} results, triggering arXiv fallback")
                
                # Execute arXiv search as fallback
                arxiv_result = self._search_arxiv_function(original_query, max_results=10)
                return arxiv_result
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error checking document search results: {e}")
            
        return None

    def generate_initial_response(self, query: str) -> str:
        """
        Generate an initial response that shows what the AI is planning to do.
        This follows the pattern of showing analysis intent using LLM.
        """
        try:
            if not self.vlm_client:
                # Fallback to simple response if LLM is not available
                return f"I'll research and analyze '{query}' to provide you with evidence-based insights and recommendations."

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

        except Exception as e:
            logger.error(f"Error generating initial response with LLM: {e}")
            # Fallback to simple response
            return f"I'll research and analyze '{query}' to provide you with evidence-based insights and recommendations."

    def process_function_calls(self, messages: List[Dict], response: Any, pdf_data: Optional[Dict] = None) -> Dict:
        """
        Process function calls from model response and generate structured response.
        Returns structured response with initial_response, sources, and final_response.
        
        Args:
            messages: List of conversation messages
            response: VLM response with potential function calls
            pdf_data: Optional PDF data containing base64 encoded PDF and filename
        """
        # Clear any leftover fallback results from previous requests
        if hasattr(self, '_arxiv_fallback_results'):
            self._arxiv_fallback_results = []
        
        # Extract the original query from messages
        original_query = ""
        try:
            for msg in messages:
                if msg.get("role") == "user":
                    original_query = msg.get("content", "")
                    break

            # Generate initial response showing intent
            initial_response = self.generate_initial_response(original_query)

            # Initialize sources list and tool calls list
            sources = []
            tool_calls_made = []

            # Add assistant message to conversation
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls if hasattr(response.choices[0].message, 'tool_calls') else None
            }
            messages.append(assistant_message)

            # Execute function calls if any
            if assistant_message.get("tool_calls"):

                for tool_call in assistant_message["tool_calls"]:
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

                        function_args["pdf_document"] = pdf_data["pdf_base64"]
                        function_args["file_name"] = pdf_data["filename"]

                        logger.info(f"Injected PDF data for summarize_pdf_document function: {pdf_data['filename']}")

                    # Execute the function
                    function_result = self.execute_function(function_name, function_args)

                    # Check if we need to trigger arXiv fallback after document search
                    if function_name == "search_document":
                        arxiv_fallback_result = self._check_and_trigger_arxiv_fallback(function_result, original_query)
                        if arxiv_fallback_result:
                            # Add arXiv tool call info for UI display
                            arxiv_tool_call_info = {
                                "function_name": "search_arxiv",
                                "function_args": {"query": original_query, "max_results": 10},
                                "display_name": self._get_tool_display_name("search_arxiv"),
                                "description": self._get_tool_description("search_arxiv", {"query": original_query, "max_results": 10})
                            }
                            tool_calls_made.append(arxiv_tool_call_info)
                            
                            # Extract sources from fallback arXiv results
                            try:
                                arxiv_data = json.loads(arxiv_fallback_result)
                                if "papers" in arxiv_data:
                                    for i, paper in enumerate(arxiv_data["papers"][:7]):
                                        source_item = {
                                            "id": f"arxiv_{i + 1}",
                                            "title": paper["title"][:100] + "..." if len(paper["title"]) > 100 else paper["title"],
                                            "url": paper["pdf_url"],
                                            "type": "ARXIV PAPER",
                                            "authors": ", ".join(paper["authors"][:3]) + (" et al." if len(paper["authors"]) > 3 else ""),
                                            "published": paper["published"]
                                        }
                                        sources.append(json.dumps(source_item))
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.error(f"Error parsing ArXiv fallback results: {e}")
                            
                            # Store fallback result for final response generation
                            if not hasattr(self, '_arxiv_fallback_results'):
                                self._arxiv_fallback_results = []
                            self._arxiv_fallback_results.append(arxiv_fallback_result)

                    # Extract sources if this is an ArXiv search
                    if function_name == "search_arxiv":
                        try:
                            result_data = json.loads(function_result)
                            if "papers" in result_data:
                                for i, paper in enumerate(
                                        result_data["papers"][:7]):  # Limit to 7 sources like Perplexity
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
                        "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else f"call_{len(messages)}"
                    })

                # Add system prompt to messages for final response
                final_messages = [{"role": "system", "content": PromptMessage.FINAL_SYSTEM_PROMPT}] + messages
                
                # Debug logging for final messages
                logger.info(f"Final messages structure:")
                for i, msg in enumerate(final_messages):
                    role = msg.get("role", "unknown")
                    content_length = len(msg.get("content", ""))
                    logger.info(f"  Message {i}: role={role}, content_length={content_length}")
                    if role == "tool":
                        logger.info(f"    Tool: {msg.get('name', 'unknown')}")
                
                # Add fallback arXiv results to the system prompt if any exist
                if hasattr(self, '_arxiv_fallback_results') and self._arxiv_fallback_results:
                    fallback_context = "\n\nADDITIONAL RESEARCH CONTEXT: The following arXiv research papers were found as supplementary information:\n"
                    for i, fallback_result in enumerate(self._arxiv_fallback_results):
                        try:
                            arxiv_data = json.loads(fallback_result)
                            if "papers" in arxiv_data:
                                for paper in arxiv_data["papers"][:3]:  # Limit to 3 papers for context
                                    fallback_context += f"- {paper.get('title', 'Unknown')}\n"
                                    fallback_context += f"  Abstract: {paper.get('summary', 'No abstract available')[:200]}...\n"
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Error processing fallback result for final response: {e}")
                    
                    # Add fallback context to the system prompt
                    final_messages[0]["content"] += fallback_context
                    
                    # Clear fallback results for next request
                    self._arxiv_fallback_results = []

                # Generate final answer with function results
                try:
                    logger.info(f"Generating final response with {len(final_messages)} messages")
                    
                    # Check total content length
                    total_content_length = sum(len(msg.get("content", "")) for msg in final_messages)
                    logger.info(f"Total content length: {total_content_length} characters")
                    
                    # Validate messages before sending
                    valid_messages = []
                    for msg in final_messages:
                        if msg.get("content") and len(msg.get("content", "").strip()) > 0:
                            valid_messages.append(msg)
                        else:
                            logger.warning(f"Skipping empty message: {msg.get('role', 'unknown')}")
                    
                    if len(valid_messages) != len(final_messages):
                        logger.warning(f"Filtered {len(final_messages) - len(valid_messages)} empty messages")
                        final_messages = valid_messages
                    
                    # If content is too long, truncate tool responses
                    if total_content_length > 32000:  # Conservative limit
                        logger.warning(f"Content too long ({total_content_length} chars), truncating tool responses")
                        for msg in final_messages:
                            if msg.get("role") == "tool" and len(msg.get("content", "")) > 2000:
                                msg["content"] = msg["content"][:2000] + "... [truncated]"
                    
                    final_response = self.vlm_client.chat.complete(
                        model=self.vlm_model,
                        messages=final_messages,
                        temperature=0.5,
                        max_tokens=1024
                    )
                    
                    final_response_content = final_response.choices[0].message.content
                    logger.info(f"Final response generated successfully: {len(final_response_content)} characters")
                    
                    return {
                        "initial_response": initial_response,
                        "sources": sources,
                        "final_response": final_response_content,
                        "tool_calls": tool_calls_made
                    }
                except Exception as e:
                    logger.error(f"Error generating final response: {e}")
                    # Try a simpler approach with just the user query and system prompt
                    try:
                        logger.info("Attempting simplified final response generation")
                        simple_messages = [
                            {"role": "system", "content": PromptMessage.FINAL_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Based on the research conducted, provide a comprehensive answer to: {original_query}"}
                        ]
                        
                        simple_response = self.vlm_client.chat.complete(
                            model=self.vlm_model,
                            messages=simple_messages,
                            temperature=0.5,
                            max_tokens=1024
                        )
                        
                        return {
                            "initial_response": initial_response,
                            "sources": sources,
                            "final_response": simple_response.choices[0].message.content,
                            "tool_calls": tool_calls_made
                        }
                    except Exception as e2:
                        logger.error(f"Error with simplified response generation: {e2}")
                        # Return a basic fallback response
                        fallback_response = f"Based on the research conducted, here are the key findings about '{original_query}':\n\n"
                        
                        # Add document sources if available
                        if sources:
                            fallback_response += "Research sources have been found and are available in the Sources tab.\n\n"
                            fallback_response += "Please check the Sources tab for detailed information from the research papers and documents."
                        else:
                            fallback_response += "While I was able to search through the available resources, I couldn't generate a comprehensive response at this time. Please try rephrasing your question or check the Sources tab for available research materials."
                        
                        return {
                            "initial_response": initial_response,
                            "sources": sources,
                            "final_response": fallback_response,
                            "tool_calls": tool_calls_made
                        }

            else:
                # No function calls, return direct response with structure
                return {
                    "initial_response": initial_response,
                    "sources": sources,
                    "final_response": response.choices[0].message.content,
                    "tool_calls": tool_calls_made
                }

        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            # Clear any leftover fallback results
            if hasattr(self, '_arxiv_fallback_results'):
                self._arxiv_fallback_results = []
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