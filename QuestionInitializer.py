"""
QuestionInitializer.py - Orchestrates the question answering workflow.

This module serves as the main entry point for the question answering system,
coordinating the various components including template management,
chain construction, and answer generation.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from pathlib import Path

from ChainManager import ChainManager
from QuestionAnswerer import QuestionAnswerer
from TemplateManager import TemplateManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuestionInitializer:
    """Orchestrates the question answering workflow including templates and pipeline execution."""
    
    def __init__(
        self,
        datastore: Any,
        model: Any,
        embedding_model: Any = None,
        embedding_type: str = "sentence_transformer",
        template_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        use_reranking: bool = True,
        save_outputs: bool = False,
        output_file_path: str = "qa_outputs.txt",
        num_responses: int = 1,
        namespace: str = "",
        save_analytics: bool = False,  # Add this
        analytics_dir: str = "retrieval_analytics"  # Add this
    ):
        """
        Initialize the question answering orchestrator.
        
        Args:
            datastore: Vector database for document retrieval
            model: Language model for text generation
            embedding_model: Model for generating text embeddings
            embedding_type: Type of embedding model ('sentence_transformer' or 'openai')
            template_path: Path to prompt templates JSON file
            ground_truth_path: Path to ground truth answers JSON file
            use_reranking: Whether to use document reranking
            save_outputs: Whether to save outputs to file
            output_file_path: Path for saving outputs
            num_responses: Number of candidate responses to generate
        """
        # Core components
        self.datastore = datastore
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        
        # Configuration paths
        self.template_path = template_path
        self.ground_truth_path = ground_truth_path
        self.save_analytics = save_analytics
        self.analytics_dir = analytics_dir
        
        # Processing options
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        self.num_responses = num_responses
        self.namespace = namespace
        
        # Initialize the template manager
        self.template_manager = TemplateManager(template_path)
        
        # These components will be initialized on first use
        self._chain_manager = None
        self._question_answerer = None

    def _initialize_pipeline(self, template: str) -> None:
        """
        Initialize chain manager and question answerer components.
        
        Args:
            template: The prompt template to use for the QA chain
        """
        logger.debug("Initializing QA pipeline components")
        
        # Create the chain manager
        self._chain_manager = ChainManager(
            datastore=self.datastore, 
            model=self.model, 
            template=template,
            namespace=self.namespace  # Pass namespace to ChainManager
        )
        
        # Setup the chain and get the retriever
        chain = self._chain_manager.setup_chain()
        chain_retriever = self._chain_manager.get_retriever()  # Get retriever for direct document access

        
        # Create the question answerer
        self._question_answerer = QuestionAnswerer(
            chain=chain,
            embedding_model=self.embedding_model,
            embedding_type=self.embedding_type,
            ground_truth_path=self.ground_truth_path,
            use_reranking=self.use_reranking,
            save_outputs=self.save_outputs,
            output_file_path=self.output_file_path,
            num_responses=self.num_responses,
            namespace=self.namespace  # Pass namespace to QuestionAnswerer
        )

    def process_questions(
        self,
        questions: List[str],
        use_ground_truth: bool = False,
        template_name: str = "default",
        save_analytics: bool = False,  
        analytics_dir: str = "retrieval_analytics" 
    ) -> Tuple[List[Dict], float]:
        """
        Process a list of questions and return results with execution time.
        
        Args:
            questions: List of questions to process
            use_ground_truth: Whether to evaluate against ground truth
            template_name: Name of the template to use
            
        Returns:
            Tuple of (results, processing_time_in_seconds)
        """
        if not questions:
            logger.warning("Empty questions list provided")
            return [], 0.0
            
        try:
            # Load template
            template = self.template_manager.get_template(template_name)
            logger.info(f"Using template: {template_name}")
            
            # Initialize pipeline components
            self._initialize_pipeline(template)
            
            # Process questions and measure time
            logger.info(f"Processing {len(questions)} questions")
            start_time = time.time()
            
            results = self._question_answerer.answer_questions(
                questions=questions,
                datastore=self.datastore,
                use_ground_truth=use_ground_truth,
                save_analytics=save_analytics,
                analytics_dir=analytics_dir
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return results, processing_time
            
        except Exception as e:
            logger.error(f"Failed to process questions: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process questions: {e}")