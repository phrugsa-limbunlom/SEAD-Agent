import arxiv
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os
import pymupdf

logger = logging.getLogger(__name__)

class ArxivService:
    """
    Service for fetching and processing research papers from arXiv
    """
    
    def __init__(self):
        self.client = arxiv.Client()
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Define research categories relevant to structural engineering and design
        self.relevant_categories = {
            "physics": ["physics.app-ph", "physics.class-ph"],  # Applied Physics, Classical Physics
            "cs": ["cs.CE", "cs.CG", "cs.CV"],  # Computational Engineering, Computational Geometry, Computer Vision
            "math": ["math.NA", "math.OC", "math.MG"],  # Numerical Analysis, Optimization, Metric Geometry
            "stat": ["stat.AP"],  # Applied Statistics
            "eess": ["eess.SP"],  # Signal Processing
            "cond-mat": ["cond-mat.mtrl-sci"]  # Materials Science
        }
    
    def search_papers(self, query: str, max_results: int = 50, 
                     sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> List[Dict]:
        """
        Search for papers on arXiv based on query
        """
        try:
            # Enhanced query with structural engineering keywords
            enhanced_query = self._enhance_query_for_structural_design(query)
            
            search = arxiv.Search(
                query=enhanced_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                paper_data = {
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "published": result.published,
                    "updated": result.updated,
                    "pdf_url": result.pdf_url,
                    "relevance_score": self._calculate_relevance_score(result, query)
                }
                papers.append(paper_data)
            
            # Sort by relevance score
            papers.sort(key=lambda x: x["relevance_score"], reverse=True)
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def get_recent_papers(self, days: int = 30, categories: List[str] = None) -> List[Dict]:
        """
        Get recent papers from specified categories
        """
        if categories is None:
            categories = [cat for cat_list in self.relevant_categories.values() for cat in cat_list]
        
        papers = []
        for category in categories:
            try:
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=20,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                for result in self.client.results(search):
                    if (datetime.now() - result.published.replace(tzinfo=None)).days <= days:
                        paper_data = {
                            "arxiv_id": result.entry_id.split("/")[-1],
                            "title": result.title,
                            "authors": [author.name for author in result.authors],
                            "abstract": result.summary,
                            "categories": result.categories,
                            "published": result.published,
                            "pdf_url": result.pdf_url,
                            "category": category
                        }
                        papers.append(paper_data)
                        
            except Exception as e:
                logger.error(f"Error fetching recent papers for {category}: {e}")
                continue
        
        return papers
    
    def _enhance_query_for_structural_design(self, query: str) -> str:
        """
        Enhance user query with structural engineering and design keywords
        """
        design_keywords = [
            "structural engineering", "structural design", "structural analysis",
            "finite element", "computational mechanics", "building design",
            "architectural design", "geometric design", "optimization",
            "materials science", "construction", "seismic design",
            "wind engineering", "concrete", "steel", "composite materials"
        ]
        
        # Add relevant keywords to improve search results
        enhanced_query = f"({query})"
        
        # Add OR conditions for relevant terms
        keyword_conditions = []
        for keyword in design_keywords:
            if keyword.lower() in query.lower():
                keyword_conditions.append(f'"{keyword}"')
        
        if keyword_conditions:
            enhanced_query += f" OR ({' OR '.join(keyword_conditions)})"
        
        return enhanced_query
    
    def _calculate_relevance_score(self, paper: arxiv.Result, query: str) -> float:
        """
        Calculate relevance score based on title, abstract, and categories
        """
        score = 0.0
        query_terms = query.lower().split()
        
        # Title relevance (higher weight)
        title_lower = paper.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        score += title_matches * 0.4
        
        # Abstract relevance
        abstract_lower = paper.summary.lower()
        abstract_matches = sum(1 for term in query_terms if term in abstract_lower)
        score += abstract_matches * 0.3
        
        # Category relevance
        relevant_cats = [cat for cat_list in self.relevant_categories.values() for cat in cat_list]
        category_matches = sum(1 for cat in paper.categories if cat in relevant_cats)
        score += category_matches * 0.2
        
        # Recency bonus (newer papers get slight boost)
        days_old = (datetime.now() - paper.published.replace(tzinfo=None)).days
        if days_old < 365:  # Papers less than a year old
            score += 0.1 * (365 - days_old) / 365
        
        return score
    
    def download_paper_text(self, arxiv_id: str) -> Optional[str]:
        """
        Download and extract text from arXiv paper PDF
        """
        try:
            # Get paper info
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search))
            
            # Download PDF
            temp_filename = f"temp_{arxiv_id}.pdf"
            paper.download_pdf(filename=temp_filename)
            
            # Extract text using pymupdf
            doc = pymupdf.open(temp_filename)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            return text
            
        except Exception as e:
            logger.error(f"Error downloading paper {arxiv_id}: {e}")
            return None 