from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Simplified document metadata for the RAG-powered structural engineering and architectural design system.
    Focuses on essential information needed for document processing and retrieval.
    """
    
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename of the document")
    file_size: int = Field(..., description="Size of the document in bytes")
    content_type: str = Field(default="application/pdf", description="MIME type of the document")
    
    # Content analysis
    title: Optional[str] = Field(None, description="Extracted or inferred document title")
    author: Optional[str] = Field(None, description="Document author(s)")
    abstract: Optional[str] = Field(None, description="Document abstract or summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords and topics")
    
    # Processing metadata
    upload_timestamp: datetime = Field(default_factory=datetime.now, description="When the document was uploaded")
    processed_timestamp: Optional[datetime] = Field(None, description="When the document processing was completed")
    processing_status: str = Field(default="pending", description="Current processing status")
    
    # Content structure
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    word_count: Optional[int] = Field(None, description="Approximate word count")
    chunk_count: Optional[int] = Field(None, description="Number of text chunks created")
    
    # Vector store information
    vector_store_id: Optional[str] = Field(None, description="ID in the vector store")
    embedding_model: Optional[str] = Field(None, description="Model used for embeddings")
    
    # Design analysis metadata
    design_categories: List[str] = Field(default_factory=list, description="Identified design categories (structural, architectural)")
    engineering_domain: Optional[str] = Field(None, description="Primary engineering domain (structural, architectural, civil)")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentSearchResult(BaseModel):
    """
    Model representing a document search result with relevance scoring.
    """
    
    document_id: str
    filename: str
    title: Optional[str]
    relevance_score: float
    matched_content: str
    chunk_ids: List[str]
    metadata: DocumentMetadata


class ResearchPaperMetadata(BaseModel):
    """
    Metadata specifically for research papers from arXiv and other sources
    """
    
    # Paper identification
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    
    # Publication info
    published_date: datetime = Field(..., description="Publication date")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    journal: Optional[str] = Field(None, description="Journal name if published")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    
    # Design relevance
    research_domain: List[str] = Field(default_factory=list, description="Research domains (structural, architectural, etc.)")
    design_applications: List[str] = Field(default_factory=list, description="Practical design applications")
    
    # Processing metadata
    processed_timestamp: Optional[datetime] = Field(None, description="When the paper was processed")
    vector_store_id: Optional[str] = Field(None, description="ID in the vector store")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DesignRecommendation(BaseModel):
    """
    Model for design recommendations generated from research papers
    """
    
    recommendation_id: str = Field(..., description="Unique identifier for the recommendation")
    recommendation_text: str = Field(..., description="The main recommendation text")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    
    # Design context
    design_domain: str = Field(..., description="structural, architectural, civil, etc.")
    application_area: str = Field(..., description="buildings, bridges, foundations, etc.")
    
    # Supporting evidence
    evidence_strength: str = Field(..., description="strong, moderate, weak")
    source_papers: List[str] = Field(default_factory=list, description="arXiv IDs of source papers")
    
    # Practical considerations
    implementation_complexity: str = Field(..., description="low, medium, high")
    safety_considerations: List[str] = Field(default_factory=list, description="Safety-related considerations")
    
    # Metadata
    generated_timestamp: datetime = Field(default_factory=datetime.now, description="When the recommendation was generated")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 