"""
main.py - Entry point for the Question Answering Pipeline.

This module integrates various components of the RAG pipeline, handles configuration,
and provides a simple interface for asking questions against a vector database.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple

# Pipeline components
from LLMProvider import create_llm_provider
from QuestionInitializer import QuestionInitializer
from ResponseFormatter import ResponseFormatter
from LLMProvider import TokenUsageCallbackHandler
from TokenTracker import TokenTracker

# Add these lines to load environment variables
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('env_variables.env')  # Specify path if needed


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAPipeline:
    """Main class to orchestrate the question answering process."""
    
    # Supported embedding models
    SUPPORTED_EMBEDDINGS = {
        "openai": ["text-embedding-3-small", "text-embedding-3-large"],
        "sentence_transformer": [
            "all-mpnet-base-v2", 
            "all-MiniLM-L6-v2",
            "multi-qa-mpnet-base-dot-v1"
        ]
    }
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",  # Updated default
        embedding_type: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        pinecone_namespace: Optional[str] = None,  # Added namespace
        use_reranking: bool = True,
        num_responses: int = 3,
        save_analytics: bool = False,
        analytics_dir: str = "retrieval_analytics",
        track_usage: bool = False
    ):
        """
        Initialize the QA Pipeline with configuration.
        
        Args:
            config_path: Optional path to JSON configuration file
            llm_provider: LLM provider ('openai' or 'ollama')
            llm_model: Model name to use for text generation
            embedding_type: Type of embedding model ('openai' or 'sentence_transformer')
            embedding_model: Name of the embedding model
            pinecone_api_key: API key for Pinecone (overrides config)
            pinecone_environment: Pinecone environment (overrides config)
            pinecone_index_name: Pinecone index name (overrides config)
            pinecone_namespace: Pinecone namespace within the index (overrides config)
            use_reranking: Whether to use document reranking
            num_responses: Number of candidate responses to generate
        """
        # Load configuration (file or defaults)
        self.config = self._load_config(config_path)

        # Initialize token tracker if enabled
        self.track_usage = track_usage
        if track_usage:
            self.token_tracker = TokenTracker()

        # Override config with explicit parameters if provided
        if llm_provider:
            self.config["llm_provider"] = llm_provider
        if llm_model:
            self.config["llm_model"] = llm_model
        if embedding_type:
            self.config["embedding_type"] = embedding_type
        if embedding_model:
            self.config["embedding_model"] = embedding_model
        if pinecone_api_key:
            self.config["pinecone"]["api_key"] = pinecone_api_key
        if pinecone_environment:
            self.config["pinecone"]["environment"] = pinecone_environment
        if pinecone_index_name:
            self.config["pinecone"]["index_name"] = pinecone_index_name
        if pinecone_namespace:
            self.config["pinecone"]["namespace"] = pinecone_namespace
        
        # Set reranking and response configuration
        self.config["use_reranking"] = use_reranking
        self.config["num_responses"] = num_responses
        self.config["save_analytics"] = save_analytics
        self.config["analytics_dir"] = analytics_dir
        
        logger.info(f"Initializing QA Pipeline with LLM: {self.config['llm_model']} and "
                  f"embedding model: {self.config['embedding_model']}")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",  # Updated default
            "embedding_type": "openai",
            "embedding_model": "text-embedding-3-small",
            "pinecone": {
                "api_key": os.environ.get("PINECONE_API_KEY", ""),
                "environment": "gcp-starter",
                "index_name": "rag-index",
                "namespace": ""  # Default empty namespace
            },
            "use_reranking": True,
            "num_responses": 3,
            "template_path": None,
            "ground_truth_path": None,
            "save_outputs": False
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Update default config with user-provided values
            merged_config = {**default_config, **user_config}
            
            # Ensure nested dicts are properly merged
            if "pinecone" in user_config:
                merged_config["pinecone"] = {**default_config["pinecone"], **user_config["pinecone"]}
                
            logger.info(f"Loaded configuration from {config_path}")
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading config, using defaults: {e}")
            return default_config
            
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check LLM provider
        if self.config["llm_provider"] not in ["openai", "ollama"]:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
            
        # Check embedding type
        if self.config["embedding_type"] not in ["openai", "sentence_transformer"]:
            raise ValueError(f"Unsupported embedding type: {self.config['embedding_type']}")
            
        # Validate embedding model matches the type
        embedding_type = self.config["embedding_type"]
        embedding_model = self.config["embedding_model"]
        
        if embedding_type in self.SUPPORTED_EMBEDDINGS:
            if embedding_model not in self.SUPPORTED_EMBEDDINGS[embedding_type]:
                logger.warning(f"Embedding model {embedding_model} not in recognized list for {embedding_type}")
                logger.warning(f"Supported models: {self.SUPPORTED_EMBEDDINGS[embedding_type]}")
        
        # Validate Pinecone configuration
        pinecone_config = self.config["pinecone"]
        if not pinecone_config["api_key"]:
            raise ValueError("Pinecone API key not provided")
        if not pinecone_config["index_name"]:
            raise ValueError("Pinecone index name not provided")

    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        try:
            # Connect to Pinecone using the new API
            pinecone_config = self.config["pinecone"]
            
            # Use the new Pinecone class instead of init()
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_config["api_key"])
            
            # Connect to the raw Pinecone index
            pinecone_index = pc.Index(pinecone_config["index_name"])
            
            # Store namespace for queries
            self.namespace = pinecone_config.get("namespace", "")
            
            logger.info(f"Connected to Pinecone index: {pinecone_config['index_name']}" + 
                    (f", namespace: {self.namespace}" if self.namespace else ""))
            
            # Create token usage tracker
            self.token_handler = TokenUsageCallbackHandler()

            # Initialize LLM with callback - This section was changed from the original code 
            if self.config["llm_provider"] == "openai":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model_name=self.config["llm_model"],
                    temperature=0.3,
                    callbacks=[self.token_handler]
                )
            else:  # "ollama"
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=self.config["llm_model"],
                    callbacks=[self.token_handler]
                )
            
            # Initialize embedding model
            if self.config["embedding_type"] == "openai":
                from langchain_openai.embeddings import OpenAIEmbeddings
                self.embedding_model = OpenAIEmbeddings(
                    model=self.config["embedding_model"]
                )
            else:  # sentence_transformer
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.config["embedding_model"]
                )
            
            # Create LangChain vector store from Pinecone index
            from langchain_pinecone import PineconeVectorStore
            self.vector_store = PineconeVectorStore(
                index=pinecone_index,
                embedding=self.embedding_model,
                namespace=self.namespace
            )
            
            # Initialize formatter
            self.formatter = ResponseFormatter(debug_mode=False)
            
            # Initialize question initializer (orchestrator)
            self.question_initializer = QuestionInitializer(
                datastore=self.vector_store,
                model=self.llm,
                embedding_model=self.embedding_model,
                embedding_type=self.config["embedding_type"],
                template_path=self.config.get("template_path"),
                ground_truth_path=self.config.get("ground_truth_path"),
                use_reranking=self.config["use_reranking"],
                save_outputs=self.config.get("save_outputs", False),
                num_responses=self.config["num_responses"],
                namespace=self.namespace,
                save_analytics=self.config.get("save_analytics", False),
                analytics_dir=self.config.get("analytics_dir", "retrieval_analytics")
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize QA Pipeline: {e}")
            

    def ask(self, questions: List[str]) -> Tuple:
        """
        Process a list of questions and return answers.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of dictionaries containing answers and metadata
        """
        if not questions:
            logger.warning("Empty questions list provided")
            return []
            
        logger.info(f"Processing {len(questions)} questions")
        
        try:
            results, processing_time = self.question_initializer.process_questions(
                questions=questions,
                template_name="default",
                save_analytics=self.config.get("save_analytics", False),
                analytics_dir=self.config.get("analytics_dir", "retrieval_analytics")
            )
            
            logger.info(f"Completed in {processing_time:.2f} seconds")
            return results, processing_time
            
        except Exception as e:
            logger.error(f"Error processing questions: {e}", exc_info=True)
            return [{"question": q, "answer": f"Error: {str(e)}", "confidence": 0.0} for q in questions]
            
    def ask_one(self, question: str) -> Dict:
        """
        Process a single question and return the answer.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing the answer and metadata
        """
        results, _ = self.ask([question])
        return results[0] if results else {
            "question": question,
            "answer": "Error: Failed to process question", 
            "confidence": 0.0
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Question Answering Pipeline")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")

    # Save outputs
    parser.add_argument("--save_analytics", action="store_true", 
                      help="Save detailed retrieval analytics to Excel files")
    parser.add_argument("--analytics_dir", type=str, default="retrieval_analytics",
                      help="Directory to save retrieval analytics")
    
    # Track usage
    parser.add_argument("--track_usage", action="store_true", 
                      help="Track token usage in persistent storage")
    parser.add_argument("--usage_report", action="store_true",
                      help="Generate a token usage report")
    
    # Model selection
    parser.add_argument("--llm_provider", type=str, choices=["openai", "ollama"], 
                        help="LLM provider to use")
    parser.add_argument("--llm_model", type=str, 
                        help="Name of the language model")
    parser.add_argument("--embedding_type", type=str, choices=["openai", "sentence_transformer"], 
                        help="Type of embedding model")
    parser.add_argument("--embedding_model", type=str, 
                        help="Name of the embedding model")
    
    # Pinecone configuration
    parser.add_argument("--pinecone_index", type=str, 
                        help="Pinecone index name")
    parser.add_argument("--pinecone_namespace", type=str, 
                        help="Pinecone namespace within the index")
    
    # Questions
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--questions", type=str, nargs="+", 
                      help="Questions to ask")
    group.add_argument("--questions_file", type=str, 
                      help="Path to file containing questions (one per line)")
    
    # Add token tracking argument
    parser.add_argument("--show_token_usage", action="store_true", 
                      help="Show token usage statistics after completion")

    return parser.parse_args()

def main():
    """Main entry point for the QA Pipeline."""
    args = parse_arguments()
    
    try:
        # Load questions if provided via file
        questions = []
        if args.questions:
            questions = args.questions
        elif args.questions_file:
            with open(args.questions_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        # Create the pipeline
        pipeline = QAPipeline(
            config_path=args.config,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            embedding_type=args.embedding_type,
            embedding_model=args.embedding_model,
            pinecone_index_name=args.pinecone_index,
            pinecone_namespace=args.pinecone_namespace,
            save_analytics=args.save_analytics,
            analytics_dir=args.analytics_dir,
            track_usage=args.track_usage
        )
        
        # Interactive mode if no questions provided
        if not questions:
            print("\nQuestion Answering Pipeline - Interactive Mode")
            print("Type 'exit' to quit\n")
            
            while True:
                question = input("\nEnter your question: ")
                if question.lower() in ["exit", "quit"]:
                    break
                    
                print("\nProcessing your question...")
                result = pipeline.ask_one(question)
                print("\n" + result["answer"])
                print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
                
        # Process provided questions
        else:
            results, processing_time = pipeline.ask(questions)
            for result in results:
                print("\n" + "="*50)
                print(f"Q: {result['question']}")
                print(f"A: {result['answer']}")
                print(f"Confidence: {result.get('confidence', 0.0):.2f}")

            if args.show_token_usage and hasattr(pipeline, 'token_handler'):
                usage = pipeline.token_handler.get_summary()
                cost = pipeline.token_handler.estimate_cost()
                
                print("\n=== Token Usage Statistics ===")
                print(f"Total requests: {usage['requests_count']}")
                print(f"Input tokens: {usage['total_input_tokens']}")
                print(f"Output tokens: {usage['total_output_tokens']}")
                print(f"Total tokens: {usage['total_tokens']}")
                
                # Price adjustment based on model
                model = pipeline.config['llm_model'].lower()
                input_price = 0.01  # Default price
                output_price = 0.03  # Default price
                
                # GPT-4o pricing
                if 'gpt-4o-mini' in model:
                    input_price = 0.0015
                    output_price = 0.002
                elif 'gpt-4o' in model:
                    input_price = 0.005
                    output_price = 0.015
                # GPT-3.5 pricing
                elif 'gpt-3.5' in model:
                    input_price = 0.0005
                    output_price = 0.0015
                    
                cost = pipeline.token_handler.estimate_cost(
                    input_price_per_1k=input_price,
                    output_price_per_1k=output_price
                )
                
                print(f"Estimated cost: ${cost['total_cost']:.4f}")
                print(f"  • Input cost: ${cost['input_cost']:.4f} (${input_price}/1K tokens)")
                print(f"  • Output cost: ${cost['output_cost']:.4f} (${output_price}/1K tokens)")
            
                # Record token usage if tracking is enabled
                if args.track_usage and hasattr(pipeline, 'token_tracker'):
                    for i, result in enumerate(results):
                        question = result.get('question', 'Unknown question')
                        pipeline.token_tracker.record_usage(
                            question=question,
                            model=pipeline.config['llm_model'],
                            input_tokens=usage['total_input_tokens'] // len(results),  # Estimate per question
                            output_tokens=usage['total_output_tokens'] // len(results),  # Estimate per question
                            doc_count=getattr(result, 'doc_count', 0) or 5,  # Default to 5 if not available
                            namespace=pipeline.namespace or "default",
                            processing_time=processing_time / len(results),  # Estimate per question
                            input_price_per_1k=input_price,
                            output_price_per_1k=output_price
                        )
                    
                    print(f"\nToken usage recorded in: {pipeline.token_tracker.tracking_file}")
            
            # Generate usage report if requested
            if args.usage_report and hasattr(pipeline, 'token_tracker'):
                report_path = pipeline.token_tracker.generate_report()
                print(f"\nUsage report generated: {report_path}")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()