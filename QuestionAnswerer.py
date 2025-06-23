"""
QuestionAnswerer.py - Core RAG question answering implementation.

This module handles the core question answering functionality, including document
retrieval, reranking, response generation, and response selection.
"""

import time
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any

from ResponseSelector import ResponseSelector
from ResponseFormatter import ResponseFormatter, QAResponse
from ScoringMetric import ScoringMetric

# Setup logging
logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """Manages the question answering workflow from retrieval to response generation."""
    
    # Constants
    EXCLUDED_DOCUMENT_STARTS = (
        "summary", 
        "overview",
        "engineering duty officer school basic and reserve courses"
    )
    
    def __init__(
        self, 
        chain: Any,
        embedding_model: Any, 
        embedding_type: str,
        chain_retriever: Any = None,
        ground_truth_path: Optional[str] = None, 
        use_reranking: bool = True, 
        save_outputs: bool = False, 
        output_file_path: str = "reranking_outputs.txt",
        num_responses: int = 5,
        max_retrieved_docs: int = 30,
        num_reranked_docs: int = 2,
        namespace: str = ""
    ):
        """
        Initialize the QuestionAnswerer with chain, embeddings, and configuration.
        
        Args:
            chain: LangChain chain for generating responses
            embedding_model: Model for generating text embeddings 
            embedding_type: Type of embeddings ('sentence_transformer' or 'openai')
            chain_retriever: Optional separate retriever from the chain
            ground_truth_path: Path to ground truth answers for evaluation
            use_reranking: Whether to use document reranking
            save_outputs: Whether to save detailed outputs to file
            output_file_path: Path for saving detailed outputs
            num_responses: Number of candidate responses to generate
            max_retrieved_docs: Maximum documents to retrieve from vector store
            num_reranked_docs: Number of top reranked docs to use
        """
        # Core components
        self.chain = chain
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.chain_retriever = chain_retriever
        
        # Configuration
        self.ground_truth_path = ground_truth_path
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        self.num_responses = num_responses
        self.max_retrieved_docs = max_retrieved_docs
        self.num_reranked_docs = num_reranked_docs
        self.namespace = namespace
        
        # Load ground truth data if available
        self.ground_truth = self._load_ground_truth()
        
        logger.info(f"QuestionAnswerer initialized with {embedding_type} embeddings")

    def _load_ground_truth(self) -> Dict[str, str]:
        """
        Load ground truth data from JSON file.
        
        Returns:
            Dictionary mapping questions to expected answers
        """
        if not self.ground_truth_path:
            return {}
            
        try:
            with open(self.ground_truth_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Loaded ground truth with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return {}

    def _is_excluded_document(self, content: str) -> bool:
        """
        Check if document should be excluded based on start text.
        
        Args:
            content: Document content to check
            
        Returns:
            True if document should be excluded, False otherwise
        """
        normalized_content = content.lower().strip()
        return any(normalized_content.startswith(prefix) for prefix in self.EXCLUDED_DOCUMENT_STARTS)

    def _retrieve_documents(self, question: str, datastore: Any) -> List[tuple]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            question: User question
            datastore: Vector store to query
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Check if this is Pinecone and if we have a namespace
            if hasattr(datastore, 'similarity_search_with_score') and self.namespace:
                # Retrieve documents with scores using namespace
                raw_results = datastore.similarity_search_with_score(
                    query=question, 
                    k=self.max_retrieved_docs,
                    namespace=self.namespace  # Add namespace parameter
                )
            else:
                # Standard retrieval without namespace
                raw_results = datastore.similarity_search_with_score(
                    query=question, 
                    k=self.max_retrieved_docs
                )
            
            # Filter out excluded documents
            retrieved_results = [
                result for result in raw_results 
                if not self._is_excluded_document(result[0].page_content)
            ]
            
            logger.info(f"Retrieved {len(retrieved_results)} documents after filtering")
            return retrieved_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []

    def _rerank_documents(
        self, 
        question: str, 
        retrieved_results: List[tuple]
    ) -> List[str]:
        """
        Rerank retrieved documents using the ResponseSelector.
        
        Args:
            question: User question
            retrieved_results: Documents with scores from vector search
            
        Returns:
            List of selected document contents for context
        """
        # Define minimum documents to use
        MIN_DOCS = 5
        
        selector = ResponseSelector(
            use_reranking=self.use_reranking,
            save_outputs=self.save_outputs,
            output_file_path=self.output_file_path
        )
        
        try:
            # Select documents for reranking
            k = max(self.num_reranked_docs * 5, MIN_DOCS * 2)  # Use larger pool for reranking
            top_percentage = min(0.5, max(0.3, len(retrieved_results) / k))
            top_percentage_index = int(np.ceil(len(retrieved_results) * top_percentage))
            
            # Ensure we have at least MIN_DOCS documents for reranking if available
            top_percentage_index = max(top_percentage_index, min(MIN_DOCS * 2, len(retrieved_results)))
            
            # Extract document content for reranking, continuing if we hit excluded docs
            documents_for_reranking = []
            i = 0
            while len(documents_for_reranking) < top_percentage_index and i < len(retrieved_results):
                doc_content = retrieved_results[i][0].page_content
                if not self._is_excluded_document(doc_content):
                    documents_for_reranking.append(doc_content)
                i += 1
                
            # Rerank documents and select top ones
            if documents_for_reranking:
                reranked_documents = selector.rerank_documents(question, documents_for_reranking)
                
                # Ensure we select at least MIN_DOCS documents if available
                num_to_select = max(self.num_reranked_docs, min(MIN_DOCS, len(reranked_documents)))
                top_documents = [doc for doc, _ in reranked_documents[:num_to_select]]
                
                logger.info(f"Reranked {len(documents_for_reranking)} documents, selected top {len(top_documents)}")
                return top_documents
            else:
                logger.warning("No valid documents for reranking")
                return []
                
        except Exception as e:
            logger.error(f"Error in document reranking: {e}", exc_info=True)
            # Fallback to top documents without reranking
            top_count = max(2, min(MIN_DOCS, len(retrieved_results)))
            return [result[0].page_content for result in retrieved_results[:top_count]]

    def _get_chain_documents(self, question: str) -> List[str]:
        """
        Retrieve documents using the chain retriever.
        
        Args:
            question: User question
            
        Returns:
            List of document contents
        """
        if not self.chain_retriever:
            return []
            
        try:
            # Get documents from chain retriever
            chain_docs = self.chain_retriever.get_relevant_documents(question)
            
            # Filter and extract content
            filtered_docs = [
                doc.page_content for doc in chain_docs[:self.num_reranked_docs]
                if not self._is_excluded_document(doc.page_content)
            ]
            
            logger.info(f"Retrieved {len(filtered_docs)} documents from chain retriever")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving chain documents: {e}")
            return []

    def _generate_responses(self, question: str, context: List[str]) -> List[str]:
        """
        Generate multiple candidate responses using the LLM.
        
        Args:
            question: User question
            context: Context documents to use
            
        Returns:
            List of candidate responses
        """
        responses = []
        
        try:
            # Generate multiple responses for selection
            for i in range(self.num_responses):
                response = self.chain.invoke({
                    "question": question, 
                    "context": context
                })
                responses.append(response)
                logger.debug(f"Generated response {i+1}/{self.num_responses}")
                
            return responses
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            # Return what we have, or a default message
            return responses or ["I'm sorry, I couldn't generate a response."]

    def answer_questions(
        self, 
        questions: List[str], 
        datastore: Any,
        use_ground_truth: bool = False,
        save_analytics: bool = True,
        analytics_dir: str = "retrieval_analytics"
    ) -> List[Dict]:
        """
        Process questions and return answers with metadata.
        
        Args:
            questions: List of questions to answer
            datastore: Vector store for retrieval
            use_ground_truth: Whether to evaluate against ground truth
            save_analytics: Whether to save detailed retrieval analytics
            
        Returns:
            List of dictionaries with question, answer, and metadata
        """
        results = []
        start_time = time.time()
        
        # Initialize helpers
        selector = ResponseSelector(
            use_reranking=self.use_reranking, 
            save_outputs=self.save_outputs, 
            output_file_path=self.output_file_path
        )
        scoring_metric = ScoringMetric(self.embedding_model, self.embedding_type)
        formatter = ResponseFormatter()

        for i, question in enumerate(questions):
            question_start_time = time.time()
            logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
            
            try:
                # Step 1: Retrieve documents from vector store
                retrieved_results = self._retrieve_documents(question, datastore)
                if not retrieved_results:
                    logger.warning(f"No documents retrieved for question: {question}")
                    continue
                
                # Step 2: Rerank documents
                top_documents = self._rerank_documents(question, retrieved_results)
                
                # Step 3: Get chain retriever documents and combine with reranked docs
                chain_doc_texts = self._get_chain_documents(question)
                
                # Step 4: Combine contexts and remove duplicates
                combined_context = list(dict.fromkeys(top_documents + chain_doc_texts))
                logger.info(f"Combined context: {len(top_documents)} reranked docs + " +
                           f"{len(chain_doc_texts)} chain docs = {len(combined_context)} unique docs")
                
                # Save analytics if enabled
                if save_analytics:
                    analytics_file = self.save_retrieval_analytics(
                        question=question,
                        retrieved_results=retrieved_results,
                        reranked_documents=top_documents,
                        selected_documents=combined_context,
                        output_dir=analytics_dir
                    )

                if not combined_context:
                    logger.warning(f"No context documents available for question: {question}")
                    continue
                
                # Step 5: Generate candidate responses
                responses = self._generate_responses(question, combined_context)
                
                # Step 6: Rank responses and select the best one
                ranked_responses = selector.rank_responses(question, responses)
                best_response = selector.select_best_response(ranked_responses) 
                confidence_score = ranked_responses[0][1] if ranked_responses else 0.0
                
                # Step 7: Calculate quality scores if ground truth is available
                expected_answer = self.ground_truth.get(question) if use_ground_truth else None
                quality_scores = scoring_metric.compute_response_quality_score(
                    best_response, expected_answer
                )
                
                # Step 8: Format the response
                qa_response = QAResponse(
                    question=question,
                    answer=best_response,
                    confidence=confidence_score,
                    references=[doc[:100] + "..." for doc in combined_context[:3]]  # Include top references
                )
                formatted_response = formatter.format_response(qa_response)
                
                # Record results
                results.append({
                    'question': question,
                    'answer': formatted_response,
                    'confidence': confidence_score,
                    'quality_scores': quality_scores,
                    'processing_time': time.time() - question_start_time,
                    'ground_truth_used': bool(expected_answer)
                })
                
                # Save detailed results if enabled
                if self.save_outputs:
                    original_results = [
                        (result[0].page_content, result[1]) 
                        for result in retrieved_results
                    ]
                    re_ranked_results = selector.rerank_documents(question, [
                        result[0].page_content for result in retrieved_results[:10]
                    ])
                    response_results = ranked_responses
                    df = selector.save_results_to_dataframe(
                        original_results, re_ranked_results, response_results
                    )
                    df.to_csv("dataframe_output.csv", index=False)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}", exc_info=True)
                results.append({
                    'question': question,
                    'answer': f"Error: Could not process question. {str(e)}",
                    'confidence': 0.0,
                    'quality_scores': {},
                    'processing_time': time.time() - question_start_time,
                    'ground_truth_used': False
                })

        logger.info(f"Processed {len(questions)} questions in {time.time() - start_time:.2f} seconds")
        return results
    

    # Add this new method to the QuestionAnswerer class
    def save_retrieval_analytics(
        self, 
        question: str, 
        retrieved_results: List[tuple],
        reranked_documents: List[str],
        selected_documents: List[str],
        output_dir: str = "retrieval_analytics"
    ) -> str:
        """
        Save detailed analytics about the document retrieval process.
        
        Args:
            question: The original query
            retrieved_results: Original retrieved documents with scores
            reranked_documents: Documents after reranking
            selected_documents: Final documents selected for LLM context
            output_dir: Directory to save analytics files
            
        Returns:
            Path to the saved analytics file
        """
        import os
        import pandas as pd
        from datetime import datetime
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a timestamp and sanitized question for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_q = "".join(c if c.isalnum() else "_" for c in question[:30])
            filename = f"{timestamp}_{sanitized_q}.xlsx"
            filepath = os.path.join(output_dir, filename)
            
            # Create DataFrames for each stage of retrieval
            
            # Initial retrieval
            initial_data = []
            for i, (doc, score) in enumerate(retrieved_results):
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                else:
                    source = 'Unknown'
                    
                initial_data.append({
                    'Rank': i+1,
                    'Score': round(score, 4),
                    'Source': source,
                    'Content': doc.page_content[:500] + ('...' if len(doc.page_content) > 500 else '')
                })
            
            initial_df = pd.DataFrame(initial_data)
            
            # Reranked documents
            reranked_data = []
            for i, doc in enumerate(reranked_documents):
                reranked_data.append({
                    'Rank': i+1,
                    'Content': doc[:500] + ('...' if len(doc) > 500 else '')
                })
            
            reranked_df = pd.DataFrame(reranked_data)
            
            # Selected documents
            selected_data = []
            for i, doc in enumerate(selected_documents):
                selected_data.append({
                    'Rank': i+1,
                    'Content': doc[:500] + ('...' if len(doc) > 500 else '')
                })
            
            selected_df = pd.DataFrame(selected_data)
            
            # Create summary data
            summary_data = [{
                'Question': question,
                'Retrieved Count': len(retrieved_results),
                'Reranked Count': len(reranked_documents),
                'Selected Count': len(selected_documents),
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
            summary_df = pd.DataFrame(summary_data)
            
            # Save all DataFrames to an Excel file with multiple sheets
            with pd.ExcelWriter(filepath) as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                initial_df.to_excel(writer, sheet_name='Initial Retrieval', index=False)
                reranked_df.to_excel(writer, sheet_name='Reranked Documents', index=False)
                selected_df.to_excel(writer, sheet_name='Selected Documents', index=False)
            
            logger.info(f"Saved retrieval analytics to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save retrieval analytics: {e}", exc_info=True)
            return ""