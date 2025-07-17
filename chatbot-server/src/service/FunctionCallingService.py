import json
import logging
from typing import Optional, Any, List, Dict
from datetime import datetime
from service.VectorStoreService import VectorStoreService
from service.ArxivService import ArxivService
from service.DesignRecommendationService import DesignRecommendationService

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
                 vlm_client: Optional[Any] = None,
                 vlm_model: Optional[str] = None):
        
        self.arxiv_service = arxiv_service
        self.vector_service = vector_service
        self.design_recommendation_service = design_recommendation_service
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
            docs = vector_retriever.get_relevant_documents(query)
            
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

    def generate_initial_response(self, query: str) -> str:
        """
        Generate an initial response that shows what the AI is planning to do.
        This follows the PickSmart pattern of showing analysis intent.
        """
        # Determine what actions will be taken based on the query
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["search", "find", "papers", "research", "arxiv"]):
            if any(term in query_lower for term in ["recent", "latest", "new", "current"]):
                return f"I'll search for the latest research papers on '{query}' to provide you with current findings and evidence-based insights."
            else:
                return f"I'll search academic literature for research papers related to '{query}' to provide comprehensive analysis."
        
        elif any(term in query_lower for term in ["recommend", "design", "how to", "best practices", "approach"]):
            return f"I'll analyze current research and generate evidence-based design recommendations for '{query}' based on the latest findings."
        
        elif any(term in query_lower for term in ["analyze", "summarize", "explain", "what is", "how does"]):
            return f"I'll analyze the available research and provide a comprehensive explanation of '{query}' based on current literature."
        
        else:
            return f"I'll research and analyze '{query}' to provide you with evidence-based insights and recommendations."

    def process_function_calls(self, messages: List[Dict], response: Any) -> Dict:
        """
        Process function calls from model response and generate structured response.
        Returns structured response with initial_response, sources, and final_response.
        """
        try:
            # Extract the original query from messages
            original_query = ""
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
            messages.append(response.choices[0].message)
            
            # Execute function calls if any
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Executing function: {function_name} with args: {function_args}")
                    
                    # Track the tool call for UI display
                    tool_call_info = {
                        "function_name": function_name,
                        "function_args": function_args,
                        "display_name": self._get_tool_display_name(function_name),
                        "description": self._get_tool_description(function_name, function_args)
                    }
                    tool_calls_made.append(tool_call_info)
                    
                    # Execute the function
                    function_result = self.execute_function(function_name, function_args)
                    
                    # Extract sources if this is an ArXiv search
                    if function_name == "search_arxiv":
                        try:
                            result_data = json.loads(function_result)
                            if "papers" in result_data:
                                for i, paper in enumerate(result_data["papers"][:7]):  # Limit to 7 sources like Perplexity
                                    source_item = {
                                        "id": f"arxiv_{i+1}",
                                        "title": paper["title"][:100] + "..." if len(paper["title"]) > 100 else paper["title"],
                                        "url": paper["pdf_url"],
                                        "type": "ARXIV PAPER",
                                        "authors": ", ".join(paper["authors"][:3]) + (" et al." if len(paper["authors"]) > 3 else ""),
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
                
                # Check if client is available before making another call
                if not self.vlm_client:
                    return {
                        "initial_response": initial_response,
                        "sources": sources,
                        "final_response": "Error: Mistral client is not initialized for final response generation.",
                        "tool_calls": tool_calls_made
                    }
                
                # Enhanced system prompt for final response generation
                final_system_prompt = """
                Based on the function call results, provide a comprehensive, well-structured final analysis. 

                IMPORTANT:
                - Do NOT list individual papers or raw search results
                - Synthesize findings into clear, actionable insights
                - Provide evidence-based recommendations
                - Keep response clean and professional (sources are handled separately)
                - Focus on practical applications and key takeaways
                - Structure your response with clear sections if appropriate
                
                Your response should be informative, actionable, and directly address the user's query.
                """
                
                # Add system prompt to messages for final response
                final_messages = [{"role": "system", "content": final_system_prompt}] + messages
                
                # Generate final answer with function results
                final_response = self.vlm_client.chat.complete(
                    model=self.vlm_model,
                    messages=final_messages,
                    temperature=0.5,
                    max_tokens=1024
                )
                
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
                    "sources": sources,
                    "final_response": response.choices[0].message.content,
                    "tool_calls": tool_calls_made
                }
                
        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            return {
                "initial_response": f"I'll analyze your query: {original_query}",
                "sources": [],
                "final_response": f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question.",
                "tool_calls": []
            }

    def _get_tool_display_name(self, function_name: str) -> str:
        """
        Get user-friendly display name for tool calls
        """
        display_names = {
            "search_arxiv": "ArXiv Search",
            "search_document": "Document Search", 
            "get_design_recommendations": "Design Recommendations"
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