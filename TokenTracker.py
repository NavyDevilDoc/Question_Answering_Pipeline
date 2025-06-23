"""
TokenTracker.py - Tracks and analyzes token usage for LLM queries.

This module provides functionality to record token usage statistics for
analysis, cost projection, and optimization.
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Setup logging
import logging
logger = logging.getLogger(__name__)


class TokenTracker:
    """Records and analyzes token usage across multiple sessions."""
    
    def __init__(
        self,
        tracking_file: str = "token_usage_history.csv",
        summary_file: str = "token_usage_summary.json",
        tracking_dir: str = "usage_analytics"
    ):
        """
        Initialize the token tracker.
        
        Args:
            tracking_file: CSV file to store individual query token usage
            summary_file: JSON file to store running totals and summaries
            tracking_dir: Directory to store tracking files
        """
        self.tracking_dir = tracking_dir
        self.tracking_file = os.path.join(tracking_dir, tracking_file)
        self.summary_file = os.path.join(tracking_dir, summary_file)
        
        # Ensure directory exists
        os.makedirs(tracking_dir, exist_ok=True)
        
        # Load summary data if it exists
        self.summary = self._load_summary()
        
        # Define CSV columns
        self.columns = [
            'timestamp', 'query_id', 'question', 'model', 
            'input_tokens', 'output_tokens', 'total_tokens',
            'doc_count', 'namespace', 'estimated_cost', 
            'processing_time', 'session_id'
        ]
        
        # Create tracking file if it doesn't exist
        if not os.path.exists(self.tracking_file):
            self._initialize_tracking_file()
    
    def _initialize_tracking_file(self) -> None:
        """Create the tracking CSV file with headers."""
        try:
            with open(self.tracking_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
        except Exception as e:
            logger.error(f"Error initializing tracking file: {e}")
    
    def _load_summary(self) -> Dict:
        """Load the summary JSON file or create default."""
        default_summary = {
            'total_queries': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_processing_time': 0.0,
            'first_query_date': None,
            'last_query_date': None,
            'models_used': {},
            'namespaces_used': {},
            'avg_tokens_per_query': 0,
            'avg_cost_per_query': 0.0,
            'avg_processing_time': 0.0
        }
        
        try:
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
            return default_summary
        except Exception as e:
            logger.error(f"Error loading summary file: {e}")
            return default_summary
    
    def _save_summary(self) -> None:
        """Save the current summary to the JSON file."""
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(self.summary, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving summary file: {e}")
    
    def record_usage(
        self,
        question: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        doc_count: int,
        namespace: str,
        processing_time: float,
        session_id: Optional[str] = None,
        input_price_per_1k: float = 0.01,
        output_price_per_1k: float = 0.03
    ) -> str:
        """
        Record token usage for a query.
        
        Args:
            question: The query text
            model: LLM model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            doc_count: Number of documents used for context
            namespace: Pinecone namespace used
            processing_time: Time to process query in seconds
            session_id: Optional session identifier
            input_price_per_1k: Price per 1000 input tokens
            output_price_per_1k: Price per 1000 output tokens
            
        Returns:
            Query ID
        """
        timestamp = datetime.now().isoformat()
        total_tokens = input_tokens + output_tokens
        estimated_cost = (input_tokens / 1000 * input_price_per_1k) + (output_tokens / 1000 * output_price_per_1k)
        
        # Generate a query ID
        query_id = f"q{self.summary['total_queries'] + 1:06d}"
        
        # If no session ID provided, generate one
        if not session_id:
            session_id = f"s{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # Write to CSV
            with open(self.tracking_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, query_id, question[:100], model,
                    input_tokens, output_tokens, total_tokens,
                    doc_count, namespace, estimated_cost,
                    processing_time, session_id
                ])
                
            # Update summary
            self.summary['total_queries'] += 1
            self.summary['total_input_tokens'] += input_tokens
            self.summary['total_output_tokens'] += output_tokens
            self.summary['total_tokens'] += total_tokens
            self.summary['total_cost'] += estimated_cost
            self.summary['total_processing_time'] += processing_time
            
            # Update first/last query dates
            if not self.summary['first_query_date']:
                self.summary['first_query_date'] = timestamp
            self.summary['last_query_date'] = timestamp
            
            # Update model statistics
            if model not in self.summary['models_used']:
                self.summary['models_used'][model] = {
                    'queries': 0, 'total_tokens': 0, 'total_cost': 0.0
                }
            self.summary['models_used'][model]['queries'] += 1
            self.summary['models_used'][model]['total_tokens'] += total_tokens
            self.summary['models_used'][model]['total_cost'] += estimated_cost
            
            # Update namespace statistics
            if namespace not in self.summary['namespaces_used']:
                self.summary['namespaces_used'][namespace] = {
                    'queries': 0, 'total_tokens': 0, 'avg_doc_count': 0
                }
            ns_stats = self.summary['namespaces_used'][namespace]
            ns_stats['queries'] += 1
            ns_stats['total_tokens'] += total_tokens
            ns_stats['avg_doc_count'] = ((ns_stats['avg_doc_count'] * (ns_stats['queries'] - 1)) + doc_count) / ns_stats['queries']
            
            # Calculate averages
            self.summary['avg_tokens_per_query'] = self.summary['total_tokens'] / self.summary['total_queries']
            self.summary['avg_cost_per_query'] = self.summary['total_cost'] / self.summary['total_queries']
            self.summary['avg_processing_time'] = self.summary['total_processing_time'] / self.summary['total_queries']
            
            # Save updated summary
            self._save_summary()
            
            return query_id
            
        except Exception as e:
            logger.error(f"Error recording token usage: {e}")
            return f"error-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def get_usage_as_dataframe(self) -> pd.DataFrame:
        """Get all recorded usage as a pandas DataFrame."""
        try:
            if os.path.exists(self.tracking_file):
                return pd.read_csv(self.tracking_file)
            return pd.DataFrame(columns=self.columns)
        except Exception as e:
            logger.error(f"Error loading usage data: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        return self.summary
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a detailed usage report.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Path to the generated report
        """
        if not output_file:
            output_file = os.path.join(self.tracking_dir, f"usage_report_{datetime.now().strftime('%Y%m%d')}.xlsx")
            
        try:
            # Load data
            df = self.get_usage_as_dataframe()
            
            # Create a writer for multiple sheets
            with pd.ExcelWriter(output_file) as writer:
                # Raw data
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Daily summary
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily = df.groupby('date').agg({
                    'total_tokens': 'sum',
                    'input_tokens': 'sum',
                    'output_tokens': 'sum',
                    'estimated_cost': 'sum',
                    'query_id': 'count'
                }).reset_index()
                daily.rename(columns={'query_id': 'queries'}, inplace=True)
                daily.to_excel(writer, sheet_name='Daily Summary', index=False)
                
                # Model summary
                model_summary = df.groupby('model').agg({
                    'total_tokens': 'sum',
                    'input_tokens': 'sum',
                    'output_tokens': 'sum',
                    'estimated_cost': 'sum',
                    'query_id': 'count',
                    'processing_time': 'mean'
                }).reset_index()
                model_summary.rename(columns={
                    'query_id': 'queries',
                    'processing_time': 'avg_processing_time'
                }, inplace=True)
                model_summary.to_excel(writer, sheet_name='By Model', index=False)
                
                # Namespace summary
                ns_summary = df.groupby('namespace').agg({
                    'total_tokens': 'sum',
                    'doc_count': 'mean',
                    'query_id': 'count',
                    'estimated_cost': 'sum'
                }).reset_index()
                ns_summary.rename(columns={
                    'query_id': 'queries',
                    'doc_count': 'avg_doc_count'
                }, inplace=True)
                ns_summary.to_excel(writer, sheet_name='By Namespace', index=False)
                
                # Summary statistics as a single-row table
                summary_df = pd.DataFrame([{
                    'Total Queries': self.summary['total_queries'],
                    'Total Tokens': self.summary['total_tokens'],
                    'Total Cost ($)': round(self.summary['total_cost'], 4),
                    'Avg. Tokens/Query': round(self.summary['avg_tokens_per_query'], 1),
                    'Avg. Cost/Query ($)': round(self.summary['avg_cost_per_query'], 4),
                    'First Query': self.summary['first_query_date'],
                    'Last Query': self.summary['last_query_date'],
                    'Total Processing Time (s)': round(self.summary['total_processing_time'], 1),
                    'Avg. Processing Time (s)': round(self.summary['avg_processing_time'], 2)
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Generated usage report at: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""