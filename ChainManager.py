"""
ChainManager.py - Manages LangChain question-answering chains.

This module provides functionality to set up and manage retrieval-augmented 
generation (RAG) chains using LangChain components.
"""

import logging
from typing import Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Setup logging
logger = logging.getLogger(__name__)

class ChainManager:
    """Manages the creation and configuration of LangChain QA chains."""
    
    def __init__(self, datastore: Any, model: Any, template: str, namespace: str = ""):
        """
        Initialize the chain manager with required components.
        
        Args:
            datastore: Vector database with as_retriever() method
            model: Language model for generating answers
            template: Prompt template string
        """
        self.datastore = datastore
        self.model = model
        self.template = template
        self.namespace = namespace

    def setup_chain(self, k: int = 4, search_type: str = "similarity"):
        """
        Set up and return the question-answering chain.
        
        Args:
            k: Number of documents to retrieve
            search_type: Search strategy to use ('similarity', 'mmr', etc.)
            
        Returns:
            A LangChain pipeline for question answering
            
        Raises:
            RuntimeError: If chain setup fails
        """
        try:
            # Create output parser
            parser = StrOutputParser()
            
            # Create prompt template from the provided template string
            prompt = PromptTemplate.from_template(self.template)
            
            # Configure the retriever
            retriever = self.datastore.as_retriever(
                search_kwargs={"k": k}, 
                search_type=search_type
            )
            
            # Build the LangChain pipeline using LCEL syntax
            chain = (
                {
                    "context": itemgetter("question") | retriever,
                    "question": itemgetter("question"),
                }
                | prompt
                | self.model
                | parser
            )
            
            logger.info(f"Chain setup complete with retriever k={k}")
            return chain
            
        except Exception as e:
            logger.error(f"Failed to set up chain: {e}", exc_info=True)
            raise RuntimeError(f"Error setting up chain: {e}")
    
    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """
        Get a standalone retriever for direct document retrieval.
        
        Args:
            k: Number of documents to retrieve
            search_type: Search strategy to use ('similarity', 'mmr', etc.)
            
        Returns:
            A configured retriever from the datastore
        """
        try:
            # Check if we should use namespace
            if self.namespace:
                # Create retriever with namespace in search_kwargs
                return self.datastore.as_retriever(
                    search_kwargs={"k": k, "namespace": self.namespace},
                    search_type=search_type
                )
            else:
                # Standard retriever without namespace
                return self.datastore.as_retriever(
                    search_kwargs={"k": k},
                    search_type=search_type
                )
                
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}", exc_info=True)
            raise RuntimeError(f"Error creating retriever: {e}")