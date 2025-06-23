"""
ResponseFormatter.py - Formats QA responses for display.

This module handles the formatting and presentation of question-answer pairs,
including terminal output formatting and file export capabilities.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from TextPreprocessor import TextPreprocessor

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class QAResponse:
    """Structure for holding question-answer pairs with metadata."""
    question: str
    answer: str
    confidence: float
    references: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    timestamp: datetime = datetime.now()

class ResponseFormatter:
    """Formats RAG responses for user-friendly output."""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the response formatter.
        
        Args:
            debug_mode: Whether to include debug information in output
        """
        self.debug_mode = debug_mode
        self.text_processor = TextPreprocessor()
        logger.debug("ResponseFormatter initialized")

    def format_response(self, qa_response: QAResponse) -> str:
        """
        Format single question-answer pair.
        
        Args:
            qa_response: QA response object to format
            
        Returns:
            Formatted string representation
        """
        if self.debug_mode:
            return self._format_debug(qa_response)
        else:
            return self._format_user(qa_response)

    def _format_user(self, qa_response: QAResponse) -> str:
        """
        Format response for end users (clean presentation).
        
        Args:
            qa_response: QA response object to format
            
        Returns:
            User-friendly formatted response
        """
        output = []
        
        # Add separator
        output.append("\n" + "="*50)
        
        # Add question
        output.append(f"\nQuestion: {qa_response.question}")
        
        # Format and add answer
        formatted_answer = self.text_processor.format_text(qa_response.answer, line_length=100)
        output.append(f"\nAnswer: {formatted_answer}")
        
        # Add confidence score
        output.append(f"\nConfidence: {qa_response.confidence:.2f}")
        
        return "\n".join(output)
    
    def _format_debug(self, qa_response: QAResponse) -> str:
        """
        Format response with debug information included.
        
        Args:
            qa_response: QA response object to format
            
        Returns:
            Debug-mode formatted response with additional information
        """
        output = []
        
        # Standard user formatting
        output.append(self._format_user(qa_response))
        
        # Add debug information
        output.append("\nDebug Information:")
        output.append(f"  - Timestamp: {qa_response.timestamp}")
        
        # Add references if available
        if qa_response.references:
            output.append("  - References:")
            for i, ref in enumerate(qa_response.references, 1):
                output.append(f"    {i}. {ref[:100]}...")
        
        # Add metadata if available
        if qa_response.metadata:
            output.append("  - Metadata:")
            for key, value in qa_response.metadata.items():
                output.append(f"    {key}: {value}")
        
        return "\n".join(output)
    
    def format_batch_responses(self, responses: List[QAResponse]) -> str:
        """
        Format multiple question-answer pairs.
        
        Args:
            responses: List of QA response objects
            
        Returns:
            Combined formatted string of all responses
        """
        if not responses:
            return "No responses to format."
            
        return "\n".join(
            self.format_response(response) for response in responses
        )

    def save_to_file(self, 
                    responses: List[QAResponse], 
                    filename: str = "qa_responses",
                    output_dir: Optional[str] = None) -> str:
        """
        Save responses to file with timestamp in specified directory.
        
        Args:
            responses: List of QA response objects to save
            filename: Base filename (timestamp will be appended)
            output_dir: Directory to save the file (defaults to current directory)
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use current directory if none specified
        if output_dir is None:
            output_dir = os.getcwd()
            
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Construct full file path
        full_path = os.path.join(
            output_dir, 
            f"{filename}_{timestamp}.txt"
        )
        
        try:
            with open(full_path, "w", encoding='utf-8') as f:
                f.write(self.format_batch_responses(responses))
            logger.info(f"Responses saved to: {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to save responses: {e}")
            raise