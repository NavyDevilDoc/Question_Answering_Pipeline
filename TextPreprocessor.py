"""
TextPreprocessor.py - Provides text cleaning and formatting utilities.

This module handles text normalization, formatting for display,
and basic NLP preprocessing tasks.
"""

import re
import logging
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """A class to preprocess text for NLP tasks and format output."""
    
    def __init__(self, download_nltk_dependencies: bool = False):
        """
        Initialize the text preprocessor with required NLTK resources.
        
        Args:
            download_nltk_dependencies: Whether to download NLTK dependencies
        """
        try:
            if download_nltk_dependencies:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                logger.info("Downloaded NLTK dependencies")
                
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.debug("TextPreprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLTK resources: {e}")
            raise

    def standardize_case(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and non-word characters."""
        return re.sub(r'[^\w\s]', '', text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by collapsing multiple spaces and trimming."""
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove stopwords from list of words."""
        return [word for word in words if word not in self.stopwords]

    def lemmatize_words(self, words: List[str]) -> List[str]:
        """Lemmatize list of words."""
        return [self.lemmatizer.lemmatize(word) for word in words]

    def preprocess(self, text: str) -> str:
        """
        Clean and preprocess text for NLP tasks.
        
        Args:
            text: Raw text to process
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        try:
            # Apply preprocessing steps
            text = self.standardize_case(text)
            text = self.remove_punctuation(text)
            text = self.normalize_whitespace(text)
            
            # Tokenize and process words
            words = text.split()
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text  # Return original text if processing fails

    def format_text(self, text: str, line_length: int = 100) -> str:
        """
        Format text by adding newlines after colons and wrapping text.
        
        Args:
            text: Text to format
            line_length: Maximum line length for wrapping
            
        Returns:
            Formatted text
        """
        if not text:
            return ""
            
        try:
            # Add newline after colons (but not in URLs)
            formatted = re.sub(r'(?<!https?):(?!\/\/)', ':\n', text)
            
            # Format text with line wrapping
            words = formatted.split(' ')
            result = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word)
                
                # Handle words that contain newlines
                if '\n' in word:
                    # Add any text before the newline to current line
                    parts = word.split('\n')
                    for i, part in enumerate(parts):
                        if i > 0:  # Start a new line
                            if current_line:
                                result.append(' '.join(current_line))
                                current_line = []
                                current_length = 0
                        
                        if part:  # Add non-empty part
                            current_line.append(part)
                            current_length = sum(len(w) for w in current_line) + len(current_line) - 1
                    continue
                
                # Normal word handling
                if current_length + word_length + (1 if current_line else 0) > line_length:
                    # Line would be too long, start a new line
                    if current_line:
                        result.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Add to current line
                    current_line.append(word)
                    current_length = sum(len(w) for w in current_line) + len(current_line) - 1
            
            # Add the last line if there's anything left
            if current_line:
                result.append(' '.join(current_line))
                
            return '\n'.join(result)
            
        except Exception as e:
            logger.error(f"Error formatting text: {e}")
            return text  # Return original text if formatting fails

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        try:
            return len(word_tokenize(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to simple word count
            return len(text.split())