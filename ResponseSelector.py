"""
ResponseSelector.py - Handles document re-ranking and response selection.

This module provides advanced document re-ranking using attention-weighted BERT
embeddings and selects the best response from multiple candidates using
a multi-criteria scoring system.
"""

import logging
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ScoringMetric import ScoringMetric

# Setup logging
logger = logging.getLogger(__name__)

class ResponseSelector:
    """
    Selects the best response from candidates and re-ranks documents using
    advanced NLP techniques.
    """
    # Constants for response scoring
    REASONING_MARKERS = ['because', 'therefore', 'thus', 'consequently', 'hence', 'however', 
                        'although', 'despite', 'since', 'given that', 'as a result']
    RELEVANCE_WEIGHT = 0.4
    CONFIDENCE_WEIGHT = 0.3
    REASONING_WEIGHT = 0.2
    COHERENCE_WEIGHT = 0.1
    
# Update the ResponseSelector.__init__ method to better handle errors

    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2", 
        top_results: int = 15, 
        use_reranking: bool = True, 
        save_outputs: bool = False, 
        output_file_path: str = "re-ranking_test_outputs.txt"
    ):
        """
        Initialize response selector with embedding model and configuration.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            top_results: Number of top results to keep after ranking
            use_reranking: Whether to use document re-ranking
            save_outputs: Whether to save outputs to file
            output_file_path: Path for saving outputs
        """
        self.top_results = top_results
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        
        try:
            # Load sentence transformer model
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
            
            # Initialize scoring metric
            try:
                self.scoring_metric = ScoringMetric(model_name, 'sentence_transformer')
            except Exception as e:
                logger.warning(f"Error initializing ScoringMetric: {e}. Continuing with limited functionality.")
                # Create a very basic placeholder that just returns 0.5 for everything
                class BasicScorer:
                    def calculate_confidence(self, *args, **kwargs): return 0.5
                    def compute_relevance_score(self, *args, **kwargs): return [(doc, 0.5) for doc in args[1]]
                self.scoring_metric = BasicScorer()
            
            # Set up BERT for attention-weighted document re-ranking
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device for BERT: {self.device}")
            
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
                self.bert_model.eval()  # Set to evaluation mode
            except Exception as e:
                logger.warning(f"Failed to load BERT components: {e}. Reranking will use simpler method.")
                self.bert_model = None
                self.tokenizer = None
            
        except Exception as e:
            logger.error(f"Error initializing ResponseSelector: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize ResponseSelector: {e}")

    @lru_cache(maxsize=1000)
    def _cached_embedding(self, text: str) -> np.ndarray:
        """
        Cache embeddings for frequently seen text.
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            # Return zero vector with correct dimensions for empty text
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        return self.embedding_model.encode(text.strip())

    def _get_relevance_score(self, question: str, response: str) -> float:
        """
        Calculate semantic similarity score between question and response.
        
        Args:
            question: Question text
            response: Response text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not question or not response:
            return 0.0
            
        try:
            q_embedding = self._cached_embedding(question)
            r_embedding = self._cached_embedding(response)
            return float(util.pytorch_cos_sim(q_embedding, r_embedding))
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0

    def rank_responses(self, question: str, responses: List[str]) -> List[Tuple[str, float]]:
        """
        Rank responses by combining multiple quality criteria.
        
        Args:
            question: Question text
            responses: List of candidate responses to rank
            
        Returns:
            List of (response, score) tuples, sorted by descending score
        """
        if not responses:
            logger.warning("Empty responses list provided to rank_responses")
            return []
            
        scored_responses = []
        
        for response in responses:
            try:
                # Calculate individual component scores
                relevance = self._get_relevance_score(question, response)
                confidence = self.scoring_metric.calculate_confidence(response, question)
                
                # Check for reasoning markers - indicators of logical reasoning
                reasoning_count = sum(1 for marker in self.REASONING_MARKERS 
                                     if marker in response.lower())
                reasoning = min(1.0, reasoning_count / 3)  # Cap at 1.0
                
                # Check response coherence between sentences
                coherence = self._check_response_coherence(response)
                
                # Calculate weighted final score
                final_score = (
                    self.RELEVANCE_WEIGHT * relevance +
                    self.CONFIDENCE_WEIGHT * confidence +
                    self.REASONING_WEIGHT * reasoning +
                    self.COHERENCE_WEIGHT * coherence
                )
                
                scored_responses.append((response, final_score))
                
                logger.debug(f"Response score: {final_score:.3f} " +
                            f"(R:{relevance:.3f}/C:{confidence:.3f}/" +
                            f"R:{reasoning:.3f}/C:{coherence:.3f})")
                
            except Exception as e:
                logger.error(f"Error ranking response: {e}")
                
        # Sort and return only the top n responses
        return sorted(
            scored_responses, 
            key=lambda x: x[1], 
            reverse=True
        )[:self.top_results]

    def _check_response_coherence(self, response: str) -> float:
        """
        Check response coherence using sentence embeddings.
        
        Args:
            response: Response text
            
        Returns:
            Coherence score between 0 and 1
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0  # By default, single sentences are coherent
        
        try:
            # Calculate embeddings for each sentence
            embeddings = [self._cached_embedding(sent) for sent in sentences]
            
            # Calculate similarities between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = util.pytorch_cos_sim(embeddings[i], embeddings[i+1])
                similarities.append(float(sim))
            
            # Return average similarity
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.error(f"Error checking response coherence: {e}")
            return 0.5  # Default middle value

    def _split_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into chunks of max_length tokens.
        
        Args:
            text: Text to split
            max_length: Maximum chunk size
            
        Returns:
            List of text chunks
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_length, 
                chunk_overlap=50,
                length_function=len
            )
            chunks = splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            # Fall back to simple splitting if sophisticated splitting fails
            return [text[i:i+max_length] for i in range(0, len(text), max_length)]

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using BERT.
        
        Args:
            text: Text to encode
            
        Returns:
            Tensor containing BERT embedding
        """
        if not text or not text.strip():
            # Return zero vector with correct dimensions
            return torch.zeros((1, 768), device=self.device)
            
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                
                outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
                
        except Exception as e:
            logger.error(f"Error encoding text with BERT: {e}")
            # Return zero vector with correct dimensions on error
            return torch.zeros((1, 768), device=self.device)

    def _attention_weighted_mean(
        self, 
        chunk_embeddings: torch.Tensor, 
        query_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention-weighted mean of chunk embeddings based on query relevance.
        
        Args:
            chunk_embeddings: Tensor of shape [num_chunks, embedding_dim]
            query_embedding: Tensor of shape [1, embedding_dim]
            
        Returns:
            Attention-weighted combined embedding
        """
        try:
            # Calculate attention scores using scaled dot product attention
            scaling_factor = torch.sqrt(
                torch.tensor(chunk_embeddings.size(-1), dtype=torch.float32, device=self.device)
            ) + 1e-6
            
            attention_scores = torch.matmul(chunk_embeddings, query_embedding.T) / scaling_factor
            
            # Apply softmax to get attention weights
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=0)
            
            # Calculate weighted mean
            weighted_embeddings = chunk_embeddings * attention_weights
            return torch.sum(weighted_embeddings, dim=0)
            
        except Exception as e:
            logger.error(f"Error calculating attention-weighted mean: {e}")
            # Return mean embedding as fallback
            return torch.mean(chunk_embeddings, dim=0)

    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two tensors.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            Cosine similarity score
        """
        try:
            return float(util.pytorch_cos_sim(tensor1, tensor2).item())
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0

    def rerank_documents(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using attention-weighted BERT embeddings.
        
        Args:
            query: Query text
            documents: List of documents to re-rank
            
        Returns:
            List of (document, score) tuples, sorted by descending score
        """
        if not self.use_reranking:
            # If reranking is disabled, use simple embedding similarity
            scored_docs = [
                (doc, self._get_relevance_score(query, doc[:1000]))  # Use first 1000 chars for speed
                for doc in documents
            ]
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
        if not documents:
            logger.warning("Empty documents list provided for reranking")
            return []

        BATCH_SIZE = 8  # Process documents in batches for efficiency
        
        try:
            # Encode query
            query_embedding = self._encode_text(query)
            
            # Process documents in batches
            ranked_docs = []
            for i in range(0, len(documents), BATCH_SIZE):
                batch_docs = documents[i:i + BATCH_SIZE]
                logger.debug(f"Processing reranking batch {i//BATCH_SIZE + 1}/{len(documents)//BATCH_SIZE + 1}")
                
                # Process each document in the batch
                for doc in batch_docs:
                    # Split document into chunks
                    chunks = self._split_into_chunks(doc)
                    if not chunks:
                        continue
                        
                    # Process chunks in sub-batches
                    all_chunk_embeddings = []
                    for j in range(0, len(chunks), BATCH_SIZE):
                        sub_batch = chunks[j:j + BATCH_SIZE]
                        
                        # Encode each chunk
                        with torch.no_grad():
                            chunk_embeddings = torch.cat([
                                self._encode_text(chunk) for chunk in sub_batch
                            ])
                            all_chunk_embeddings.append(chunk_embeddings)
                    
                    # Combine all chunk embeddings
                    if all_chunk_embeddings:
                        all_chunks = torch.cat(all_chunk_embeddings)
                        
                        # Get attention-weighted document embedding
                        doc_embedding = self._attention_weighted_mean(all_chunks, query_embedding)
                        
                        # Calculate similarity
                        similarity = self._cosine_similarity(query_embedding, doc_embedding.unsqueeze(0))
                        ranked_docs.append((doc, similarity))
            
            return sorted(ranked_docs, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}", exc_info=True)
            # Fall back to basic scoring without BERT
            return [(doc, self._get_relevance_score(query, doc[:500])) for doc in documents]

    def select_best_response(self, ranked_responses: List[Tuple[str, float]]) -> str:
        """
        Select single best response from pre-ranked candidates.
        
        Args:
            ranked_responses: List of (response, score) tuples
            
        Returns:
            Best response text
        """
        if not ranked_responses:
            logger.warning("No responses available for selection")
            return "I don't have enough information to answer that question."
            
        return ranked_responses[0][0]

    def save_outputs_to_file(
        self, 
        original_outputs: List[str], 
        reranked_outputs: List[str]
    ) -> None:
        """
        Save original and re-ranked outputs to a text file.
        
        Args:
            original_outputs: Original documents/responses
            reranked_outputs: Re-ranked documents/responses
        """
        if not self.save_outputs:
            return
            
        try:
            output_path = Path(self.output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("Original Outputs:\n" + "="*50 + "\n")
                for i, output in enumerate(original_outputs, 1):
                    file.write(f"{i}. {output[:500]}...\n\n")
                    
                file.write("\nRe-ranked Outputs:\n" + "="*50 + "\n")
                for i, output in enumerate(reranked_outputs, 1):
                    file.write(f"{i}. {output[:500]}...\n\n")
                    
            logger.info(f"Saved outputs to {self.output_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save outputs: {e}")

    def save_results_to_dataframe(
        self, 
        original_results: List[Tuple[str, float]], 
        re_ranked_results: List[Tuple[str, float]], 
        responses: List[Tuple[str, float]]
    ) -> pd.DataFrame:
        """
        Save original, re-ranked results, and responses to a DataFrame.
        
        Args:
            original_results: Original retrieval results with scores
            re_ranked_results: Re-ranked results with scores
            responses: Generated responses with scores
            
        Returns:
            DataFrame containing all results
        """
        try:
            # Create a copy to avoid modifying the original lists
            orig = original_results.copy() if original_results else []
            reranked = re_ranked_results.copy() if re_ranked_results else []
            resp = responses.copy() if responses else []
            
            # Find maximum length
            max_len = max(len(orig), len(reranked), len(resp))
            
            # Pad lists to equal length
            orig.extend([("", None)] * (max_len - len(orig)))
            reranked.extend([("", None)] * (max_len - len(reranked)))
            resp.extend([("", None)] * (max_len - len(resp)))
            
            # Create DataFrame
            data = {
                "Original Result": [r[0][:500] for r in orig],  # Truncate for readability
                "Original Score": [r[1] for r in orig],
                "Re-ranked Result": [r[0][:500] for r in reranked],  # Truncate for readability
                "Re-ranked Score": [r[1] for r in reranked],
                "Response": [r[0][:500] for r in resp],  # Truncate for readability
                "Response Score": [r[1] for r in resp]
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Created DataFrame with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            # Return an empty DataFrame as fallback
            return pd.DataFrame()