from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class DocumentChunk:
    content: str
    chunk_type: str
    page_number: int
    chunk_id: str
    embedding: np.ndarray = None
    metadata: dict = None


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
    research_domain: List[str] = Field(default_factory=list,
                                       description="Research domains (structural, architectural, etc.)")
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
    generated_timestamp: datetime = Field(default_factory=datetime.now,
                                          description="When the recommendation was generated")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }