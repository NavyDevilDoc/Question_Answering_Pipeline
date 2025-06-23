"""
TemplateManager.py - Manages prompt templates for the QA system.

This module handles loading, storing, and retrieving prompt templates
from files or providing default templates when necessary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages prompt templates for the question answering system."""
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the TemplateManager with templates from a file or defaults.
        
        Args:
            template_path: Optional path to a JSON file containing templates
        """
        self.template_path = template_path
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """
        Load templates from a JSON file or set default templates.
        
        Returns:
            Dictionary of template name to template string mappings
        """
        if not self.template_path:
            logger.info("No template path provided, using default templates")
            return {"default": self._default_template()}
            
        try:
            template_file = Path(self.template_path)
            if not template_file.exists():
                logger.warning(f"Template file not found: {self.template_path}")
                return {"default": self._default_template()}
                
            with open(template_file, "r", encoding="utf-8") as file:
                templates = json.load(file)
            logger.info(f"Successfully loaded templates from {self.template_path}")
            return templates
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in template file: {self.template_path}")
            return {"default": self._default_template()}
        except Exception as e:
            logger.error(f"Failed to load templates: {e}", exc_info=True)
            return {"default": self._default_template()}

    def _default_template(self) -> str:
        """
        Provide a default template for QA tasks.
        
        Returns:
            A string containing the default prompt template
        """
        return """
        Answer the following question based on the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """

    def get_template(self, template_name: str = "default") -> str:
        """
        Retrieve a specific template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template string for the requested template or the default template
        """
        if template_name in self.templates:
            return self.templates[template_name]
        else:
            logger.warning(f"Template '{template_name}' not found. Using default template.")
            return self.templates.get("default", self._default_template())

    def add_template(self, name: str, template: str) -> None:
        """
        Add or update a template.
        
        Args:
            name: Name of the template to add or update
            template: Template string
        """
        self.templates[name] = template
        logger.debug(f"Added/updated template: {name}")

    def list_templates(self) -> list:
        """
        List all available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())