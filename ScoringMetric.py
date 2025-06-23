"""
ScoringMetric.py - Evaluates document relevance and response quality.

This module provides metrics for evaluating the quality of retrieved documents
and generated responses, with emphasis on reference-free evaluation methods
that don't require ground truth.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from functools import lru_cache

# NLP and embedding tools
from sentence_transformers import util, SentenceTransformer
from langchain_openai.embeddings import OpenAIEmbeddings
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logger = logging.getLogger(__name__)

# Flag for whether reference-based metrics are available
REFERENCE_METRICS_AVAILABLE = False

# Try to import evaluate library and its dependencies, but don't fail if not available
try:
    import evaluate
    try:
        # Check if we have absl and other dependencies by loading rouge
        rouge = evaluate.load('rouge')
        REFERENCE_METRICS_AVAILABLE = True
        logger.info("Reference-based metrics (ROUGE, BLEU) available")
    except ImportError as e:
        logger.warning(f"Reference metrics require additional dependencies: {str(e)}")
        logger.warning("To enable ROUGE and BLEU metrics, install: pip install absl-py")
except ImportError:
    logger.warning("evaluate library not found, reference-based metrics will be unavailable")
    logger.warning("To enable reference metrics, install: pip install evaluate absl-py")

class ScoringMetric:
    """Evaluates document relevance and response quality using various metrics."""
    
    # Constants for response evaluation
    REASONING_MARKERS = [
        'because', 'therefore', 'thus', 'consequently', 'since',
        'given that', 'as a result', 'hence', 'due to', 'for this reason',
        'so', 'accordingly', 'however', 'although', 'despite', 
        'even though', 'whereas', 'while', 'in contrast', 'on the other hand'
    ]
    
    def __init__(
        self, 
        embedding_model: Any, 
        embedding_type: str = "sentence_transformer"
    ):
        """
        Initialize the ScoringMetric class with embedding model and NLP tools.
        
        Args:
            embedding_model: Model or model name for generating embeddings
            embedding_type: Type of embedding model ('sentence_transformer' or 'openai')
        """
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.reference_metrics_available = REFERENCE_METRICS_AVAILABLE
        
        try:
            # Initialize NLP components for response evaluation
            self.nlp = spacy.load("en_core_web_sm")
            self.vectorizer = TfidfVectorizer()
            logger.info(f"Initialized ScoringMetric with {embedding_type} embeddings")
            
            # Initialize reference-based metrics if available (optional)
            if REFERENCE_METRICS_AVAILABLE:
                self.rouge = evaluate.load('rouge')
                self.bleu = evaluate.load('bleu')
                logger.debug("Loaded reference-based metrics (ROUGE, BLEU)")
            
        except Exception as e:
            if "en_core_web_sm" in str(e):
                logger.warning("SpaCy model not found. To install: python -m spacy download en_core_web_sm")
                # Try to load a smaller model instead
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    logger.warning("Using blank SpaCy model instead")
                    self.nlp = spacy.blank("en")
            else:
                logger.error(f"Error initializing ScoringMetric: {e}", exc_info=True)
                # Don't raise here - we want the class to still work with reduced functionality

    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching for performance.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if self.embedding_type == 'sentence_transformer':
                # Ensure we have a model instance, not just a name
                if isinstance(self.embedding_model, str):
                    self.embedding_model = SentenceTransformer(self.embedding_model)
                return self.embedding_model.encode(text)
                
            elif self.embedding_type == 'openai':
                # Ensure we have an embeddings instance
                if isinstance(self.embedding_model, str):
                    self.embedding_model = OpenAIEmbeddings(model=self.embedding_model)
                return self.embedding_model.embed_query(text)
                
            else:
                logger.warning(f"Unsupported embedding type: {self.embedding_type}")
                return np.zeros(768)  # Return zero vector as fallback
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(768)  # Return zero vector on error

    def compute_relevance_score(
        self, 
        query: str, 
        retrieved_documents: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Compute relevance scores for retrieved documents.
        
        Args:
            query: User query
            retrieved_documents: List of documents to score
            
        Returns:
            List of (document, score) tuples sorted by descending score
        """
        scored_documents = []
        
        if not query or not retrieved_documents:
            logger.warning("Empty query or document list provided to compute_relevance_score")
            return scored_documents
        
        try:
            logger.debug(f"Computing relevance for {len(retrieved_documents)} documents")
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Score each document
            for doc in retrieved_documents:
                try:
                    doc_embedding = self._get_embedding(doc)
                    similarity = float(util.pytorch_cos_sim(query_embedding, doc_embedding))
                    scored_documents.append((doc, similarity))
                except Exception as e:
                    logger.error(f"Error scoring individual document: {e}")
                    scored_documents.append((doc, 0.0))
            
            # Sort by descending score
            return sorted(scored_documents, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in relevance scoring: {e}", exc_info=True)
            return scored_documents

    def compute_response_quality_score(
        self, 
        response: str, 
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute quality scores for the generated response.
        
        Uses reference-free metrics by default, but can use reference-based metrics
        (ROUGE, BLEU) if ground_truth is provided and metrics are available.
        
        Args:
            response: Generated response to evaluate
            ground_truth: Optional reference answer for comparison
            
        Returns:
            Dictionary of quality metrics
        """
        scores = {}
        
        if not response:
            logger.warning("Empty response provided to compute_response_quality_score")
            return {"confidence": 0.0}
        
        try:
            # Calculate reference-free confidence score
            confidence_score = self.calculate_confidence(response)
            scores['confidence'] = confidence_score
            
            # Calculate linguistic features
            scores['specificity'] = self._calculate_specificity(response)
            scores['structure'] = self._calculate_structure_score(response)
            scores['coherence'] = self._calculate_coherence(response)
            
            # Calculate reference-based metrics if ground truth is provided and metrics available
            if ground_truth and self.reference_metrics_available:
                try:
                    # ROUGE scores (precision, recall, F1 for n-grams)
                    rouge_scores = self.rouge.compute(predictions=[response], references=[ground_truth])
                    scores.update({f"rouge_{k}": v for k, v in rouge_scores.items()})
                    
                    # BLEU score (n-gram precision)
                    bleu_scores = self.bleu.compute(
                        predictions=[response.split()], 
                        references=[[ground_truth.split()]]
                    )
                    scores['bleu'] = bleu_scores['bleu']
                    
                except Exception as e:
                    logger.error(f"Error computing reference-based metrics: {e}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error computing response quality score: {e}", exc_info=True)
            return {"confidence": 0.5}  # Return default confidence on error

    def calculate_confidence(
        self, 
        response: str, 
        question: Optional[str] = None
    ) -> float:
        """
        Calculate confidence score based on response characteristics.
        
        This reference-free metric combines multiple factors to estimate response quality.
        
        Args:
            response: Response to evaluate
            question: Optional question for context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not response or not response.strip():
            return 0.0
            
        try:
            # Length factor: Rewards reasonably long responses up to a point
            words = response.split()
            length_factor = min(len(words) / 100, 1.0)
            
            # Specificity: Rewards lexical diversity
            specificity = self._calculate_specificity(response)
            
            # Structure: Rewards reasoning markers
            structure = self._calculate_structure_score(response)
            
            # Coherence: Uses SpaCy for linguistic coherence
            coherence = self._calculate_coherence(response)
            
            # Informativeness: Uses TF-IDF to measure information content
            informativeness = self._calculate_informativeness(response, question)
            
            # Combine factors
            factors = {
                'length': length_factor,
                'specificity': specificity,
                'structure': structure,
                'coherence': coherence,
                'informativeness': informativeness
            }
            
            # Calculate weighted average
            return np.mean(list(factors.values()))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Return middle value on error


    def _calculate_specificity(self, response: str) -> float:
        """
        Calculate lexical diversity as ratio of unique words to total words.
        
        Args:
            response: Response text
            
        Returns:
            Specificity score between 0.0 and 1.0
        """
        words = response.lower().split()
        if not words:
            return 0.0
            
        unique_words = set(words)
        return len(unique_words) / len(words)

    def _calculate_structure_score(self, response: str) -> float:
        """
        Calculate structure score based on presence of reasoning markers.
        
        Args:
            response: Response text
            
        Returns:
            Structure score between 0.0 and 1.0
        """
        response_lower = response.lower()
        marker_count = sum(1 for marker in self.REASONING_MARKERS if marker in response_lower)
        return min(marker_count / 3, 1.0)  # Cap at 1.0

    def _calculate_coherence(self, response: str) -> float:
        """
        Calculate coherence using SpaCy linguistic features.
        
        Args:
            response: Response text
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        try:
            doc = self.nlp(response)
            
            # Check if response has multiple sentences
            sentences = list(doc.sents)
            if len(sentences) <= 1:
                return 1.0  # Single sentences are coherent by default
                
            # Check for coreference and logical connectives
            has_pronouns = any(token.pos_ == "PRON" for token in doc)
            has_connectives = any(token.dep_ in ["mark", "cc"] for token in doc)
            
            # Check for sentence transitions
            coherence_score = 0.7  # Base score
            
            if has_pronouns:
                coherence_score += 0.15
                
            if has_connectives:
                coherence_score += 0.15
                
            return min(coherence_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.7  # Return default value on error

    def _calculate_informativeness(
        self, 
        response: str, 
        question: Optional[str] = None
    ) -> float:
        """
        Calculate informativeness using TF-IDF to measure information content.
        
        Args:
            response: Response text
            question: Optional question for context
            
        Returns:
            Informativeness score between 0.0 and 1.0
        """
        try:
            if not question:
                # If no question is provided, use word count as a simple proxy
                word_count = len(response.split())
                return min(word_count / 50, 1.0)
                
            # Use question and response for TF-IDF
            documents = [question, response]
            try:
                tfidf_matrix = self.vectorizer.fit_transform(documents)
                response_tfidf = tfidf_matrix[1].toarray().flatten()
                
                # Calculate mean of non-zero values to avoid empty response penalty
                non_zeros = response_tfidf[response_tfidf > 0]
                if len(non_zeros) > 0:
                    return float(np.mean(non_zeros))
                return 0.0
                
            except ValueError:
                # Handle empty response
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating informativeness: {e}")
            return 0.5  # Return middle value on error

    # Extension point for future LLM-as-checker implementation
    def evaluate_with_llm(
        self, 
        response: str, 
        question: str, 
        context: Optional[List[str]] = None,
        llm_provider: Optional[Any] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Placeholder for LLM-based evaluation of response quality.
        
        This method will use an LLM to evaluate the response quality based on
        the question and retrieved context. Currently returns a placeholder,
        but can be implemented in the future.
        
        Args:
            response: Response to evaluate
            question: Original question
            context: Optional retrieved context used to generate response
            llm_provider: Optional LLM provider for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # This is a placeholder for future implementation
        logger.info("LLM-based evaluation requested but not yet implemented")
        
        # Return basic confidence score for now
        return {
            "confidence": self.calculate_confidence(response, question),
            "llm_evaluation": "Not implemented yet"
        }