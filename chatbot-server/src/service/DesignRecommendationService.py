import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

from service.ArxivService import ArxivService
from service.VectorStoreService import VectorStoreService
from data.DocumentMetadata import ResearchPaperMetadata, DesignRecommendation

logger = logging.getLogger(__name__)

class DesignRecommendationService:
    """
    Service for generating design recommendations based on research papers
    """
    
    def __init__(self, arxiv_service: ArxivService, vector_service: VectorStoreService, llm_client):
        self.arxiv_service = arxiv_service
        self.vector_service = vector_service
        self.llm_client = llm_client
        self.llm_model = "mixtral-8x7b-32768"  # Using your existing model
        
        # Design recommendation templates
        self.recommendation_templates = {
            "structural": """
            Based on the research findings, generate specific structural design recommendations:
            
            Research Context: {research_context}
            Design Query: {design_query}
            
            Provide recommendations in this format:
            1. Primary Recommendation: [specific actionable recommendation]
            2. Supporting Evidence: [key research findings that support this]
            3. Implementation Details: [how to implement this in practice]
            4. Limitations: [any limitations or constraints]
            5. Confidence Level: [high/medium/low based on research strength]
            
            Focus on practical, implementable solutions for structural engineering.
            """,
            
            "architectural": """
            Based on architectural research, provide design recommendations:
            
            Research Context: {research_context}
            Design Query: {design_query}
            
            Generate recommendations covering:
            1. Spatial Design: [space planning and layout recommendations]
            2. Material Selection: [recommended materials and their properties]
            3. Environmental Considerations: [sustainability and environmental impact]
            4. User Experience: [how design affects building occupants]
            5. Aesthetic Integration: [balancing function with form]
            
            Ensure recommendations are grounded in the research findings.
            """,
            
            "geometric": """
            Based on geometric and computational research, provide design recommendations:
            
            Research Context: {research_context}
            Design Query: {design_query}
            
            Focus on:
            1. Geometric Optimization: [optimal geometric configurations]
            2. Computational Methods: [algorithms and computational approaches]
            3. Performance Metrics: [how geometry affects structural performance]
            4. Fabrication Considerations: [manufacturing and construction aspects]
            5. Parametric Design: [parameter relationships and constraints]
            
            Emphasize quantifiable and measurable recommendations.
            """
        }
    
    def generate_recommendations(self, design_query: str, domain: str = "structural") -> List[DesignRecommendation]:
        """
        Generate design recommendations based on research papers
        """
        try:
            # Step 1: Search for relevant papers
            papers = self.arxiv_service.search_papers(design_query, max_results=30)
            
            if not papers:
                logger.warning("No papers found for query")
                return []
            
            # Step 2: Process and analyze papers
            processed_papers = self._process_papers_for_recommendations(papers, design_query)
            
            # Step 3: Generate recommendations
            recommendations = []
            
            # Group papers by research themes
            paper_groups = self._group_papers_by_theme(processed_papers)
            
            for theme, theme_papers in paper_groups.items():
                recommendation = self._generate_theme_recommendation(
                    theme, theme_papers, design_query, domain
                )
                if recommendation:
                    recommendations.append(recommendation)
            
            # Step 4: Rank recommendations by confidence
            recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _process_papers_for_recommendations(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        Process papers to extract design-relevant information
        """
        processed_papers = []
        
        for paper in papers:
            try:
                # Extract key information using LLM
                extraction_prompt = f"""
                Analyze this research paper for design recommendations:
                
                Title: {paper['title']}
                Abstract: {paper['abstract']}
                
                Extract:
                1. Key findings relevant to {query}
                2. Practical applications for design
                3. Design implications
                4. Limitations or constraints
                5. Confidence level of findings (high/medium/low)
                
                Return as JSON with these fields: key_findings, applications, implications, limitations, confidence
                """
                
                response = self._query_llm(extraction_prompt)
                
                try:
                    extracted_info = json.loads(response)
                    paper['extracted_info'] = extracted_info
                    processed_papers.append(paper)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response for paper {paper['arxiv_id']}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing paper {paper['arxiv_id']}: {e}")
                continue
        
        return processed_papers
    
    def _group_papers_by_theme(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group papers by research themes for coherent recommendations
        """
        # Use LLM to identify themes
        paper_summaries = []
        for paper in papers:
            summary = f"Title: {paper['title']}\nKey findings: {paper.get('extracted_info', {}).get('key_findings', '')}"
            paper_summaries.append(summary)
        
        theme_prompt = f"""
        Analyze these research papers and identify 3-5 main themes:
        
        {chr(10).join(paper_summaries)}
        
        Return themes as JSON list: ["theme1", "theme2", ...]
        Focus on design-relevant themes like "optimization methods", "material innovations", "computational approaches", etc.
        """
        
        try:
            response = self._query_llm(theme_prompt)
            themes = json.loads(response)
            
            # Assign papers to themes
            theme_groups = {theme: [] for theme in themes}
            
            for paper in papers:
                # Determine which theme this paper belongs to
                paper_text = f"{paper['title']} {paper['abstract']}"
                
                best_theme = None
                best_score = 0
                
                for theme in themes:
                    score = self._calculate_theme_similarity(paper_text, theme)
                    if score > best_score:
                        best_score = score
                        best_theme = theme
                
                if best_theme and best_score > 0.3:  # Threshold for relevance
                    theme_groups[best_theme].append(paper)
            
            # Remove empty themes
            return {theme: papers for theme, papers in theme_groups.items() if papers}
            
        except Exception as e:
            logger.error(f"Error grouping papers by theme: {e}")
            # Fallback: single theme
            return {"general": papers}
    
    def _generate_theme_recommendation(self, theme: str, papers: List[Dict], 
                                     query: str, domain: str) -> Optional[DesignRecommendation]:
        """
        Generate a design recommendation for a specific theme
        """
        try:
            # Combine research context from papers
            research_context = self._build_research_context(papers)
            
            # Select appropriate template
            template = self.recommendation_templates.get(domain, self.recommendation_templates["structural"])
            
            # Generate recommendation
            recommendation_prompt = template.format(
                research_context=research_context,
                design_query=query
            )
            
            recommendation_text = self._query_llm(recommendation_prompt)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(papers)
            
            # Create recommendation object
            recommendation = DesignRecommendation(
                recommendation_id=f"rec_{datetime.now().timestamp()}",
                source_papers=[paper['arxiv_id'] for paper in papers],
                recommendation_text=recommendation_text,
                confidence_score=confidence_score,
                design_domain=domain,
                application_area=theme,
                evidence_strength=self._assess_evidence_strength(papers),
                implementation_complexity=self._assess_complexity(recommendation_text)
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating theme recommendation: {e}")
            return None
    
    def _build_research_context(self, papers: List[Dict]) -> str:
        """
        Build research context from multiple papers
        """
        context_parts = []
        
        for paper in papers:
            extracted_info = paper.get('extracted_info', {})
            context_part = f"""
            Paper: {paper['title']}
            Authors: {', '.join(paper['authors'][:3])}
            Key Findings: {extracted_info.get('key_findings', 'Not available')}
            Applications: {extracted_info.get('applications', 'Not available')}
            """
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _query_llm(self, prompt: str) -> str:
        """
        Query the LLM with error handling
        """
        try:
            response = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return "Error generating response"
    
    def _calculate_confidence_score(self, papers: List[Dict]) -> float:
        """
        Calculate confidence score based on paper quality and consistency
        """
        if not papers:
            return 0.0
        
        # Factors: number of papers, recency, consistency
        paper_count_factor = min(len(papers) / 5.0, 1.0)  # Max at 5 papers
        
        # Recency factor (newer papers get higher weight)
        recency_scores = []
        for paper in papers:
            days_old = (datetime.now() - paper['published'].replace(tzinfo=None)).days
            recency_score = max(0, 1 - days_old / 1095)  # 3 years max
            recency_scores.append(recency_score)
        
        recency_factor = sum(recency_scores) / len(recency_scores)
        
        # Consistency factor (based on extracted confidence levels)
        confidence_levels = []
        for paper in papers:
            extracted_info = paper.get('extracted_info', {})
            confidence = extracted_info.get('confidence', 'medium')
            if confidence == 'high':
                confidence_levels.append(0.8)
            elif confidence == 'medium':
                confidence_levels.append(0.5)
            else:
                confidence_levels.append(0.2)
        
        consistency_factor = sum(confidence_levels) / len(confidence_levels)
        
        # Combined confidence score
        confidence_score = (paper_count_factor * 0.3 + recency_factor * 0.3 + consistency_factor * 0.4)
        
        return min(confidence_score, 1.0)
    
    def _assess_evidence_strength(self, papers: List[Dict]) -> str:
        """
        Assess overall evidence strength
        """
        if len(papers) >= 5:
            return "strong"
        elif len(papers) >= 3:
            return "moderate"
        else:
            return "weak"
    
    def _assess_complexity(self, recommendation_text: str) -> str:
        """
        Assess implementation complexity from recommendation text
        """
        complexity_indicators = {
            "high": ["complex", "advanced", "sophisticated", "specialized", "extensive"],
            "medium": ["moderate", "standard", "typical", "conventional"],
            "low": ["simple", "basic", "straightforward", "easy", "minimal"]
        }
        
        text_lower = recommendation_text.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return complexity
        
        return "medium"  # Default
    
    def _calculate_theme_similarity(self, text: str, theme: str) -> float:
        """
        Calculate similarity between text and theme (simplified)
        """
        text_lower = text.lower()
        theme_lower = theme.lower()
        
        # Simple keyword matching (can be enhanced with embeddings)
        theme_words = theme_lower.split()
        matches = sum(1 for word in theme_words if word in text_lower)
        
        return matches / len(theme_words) 