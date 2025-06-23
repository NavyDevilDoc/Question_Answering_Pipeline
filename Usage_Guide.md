# Question Answering Pipeline Usage Guide

## 1. Basic Usage
    a.  Interactive mode - asks for questions one at a time
            python main.py
    b.  Ask a specific question
            python main.py --questions "What is RAG?"
    c.  Ask multiple questions
            python main.py --questions "What is RAG?" "How does vector search work?"
    d.  Process questions from a file (one per line)
            python main.py --questions_file my_questions.txt


## 2. Using Different LLM Models
    a.  Use a specific OpenAI model
            python main.py --llm_model gpt-4o
    b.  Use an Ollama model
            python main.py --llm_provider ollama --llm_model llama2


## 3. Using Different Embedding Models
    a.  Use different OpenAI embedding model
            python main.py --embedding_model text-embedding-3-large
    b.  Use Sentence Transformer embedding model
            python main.py --embedding_type sentence_transformer --embedding_model all-mpnet-base-v2


## 4. Configuring Pinecone
    a.  Specify a different index
            python main.py --pinecone_index my-knowledge-base
    b.  Use a specific namespace within an index
            python main.py --pinecone_index my-knowledge-base --pinecone_namespace corporate-docs


## 5. Using a Configuration File
    a.  Load settings from JSON configuration file
            python main.py --config my_config.json
    b.  Example
            {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "embedding_type": "openai",
            "embedding_model": "text-embedding-3-small",
            "pinecone": {
                "index_name": "my-knowledge-base",
                "namespace": "finance-docs"
            },
            "use_reranking": true,
            "num_responses": 3
            }


## 6. Common Examples
    a.  Ask a specific question against a particular namespace
            python main.py --questions "What are our Q2 revenue projections?" --pinecone_namespace financial-reports
    b.  Process multiple questions with a specific model
            python main.py --questions_file quarterly_questions.txt --llm_model gpt-4o
    c.  Use a specific embedding model to match what was used during document ingestion
            python main.py --embedding_model text-embedding-3-small


## 7. Advanced Options
    a.  Turn off reranking for faster but potentially less accurate responses
            python main.py --config config.json --no_reranking
    b.  Save outputs for later reference
            python main.py --questions_file important_questions.txt --save_outputs
    c.  Compare to ground truth answers
            python main.py --questions_file test_questions.txt --ground_truth ground_truth.json


## 8. Token Usage Tracking and Analytics
    a.  Display token usage for the current session
            python main.py --questions "What is RAG?" --show_token_usage
    
    b.  Save detailed retrieval analytics to Excel files
            python main.py --questions "What is vector search?" --save_analytics
    
    c.  Specify a custom directory for analytics files
            python main.py --questions "What is RAG?" --save_analytics --analytics_dir "my_analytics"
    
    d.  Track token usage in persistent storage for long-term analysis
            python main.py --questions "What is RAG?" --show_token_usage --track_usage
    
    e.  Generate a comprehensive usage report of all historical queries
            python main.py --usage_report
    
    f.  Complete example with all tracking options
            python main.py --pinecone_index my-index --pinecone_namespace my-namespace 
                          --questions "What is RAG?" --show_token_usage --save_analytics 
                          --analytics_dir "analytics" --track_usage


## 9. Environment Variables

The pipeline looks for environment variables in a file called `env_variables.env` containing your:
 -  OpenAI API Key
 -  Pinecone API Key


## 10. Troubleshooting

- **Missing dependencies**: Run `pip install -r requirements.txt` or `install_packages_with_logging.bat`
- **CUDA/GPU issues**: Ensure PyTorch is installed with CUDA support
- **API key errors**: Check your environment variables file and API key values
- **Empty responses**: Verify the index name and namespace match where your documents were uploaded
- **Slow responses**: Consider using a smaller/faster model or disabling reranking for quick testing

## 11. Default Settings

- **LLM**: OpenAI `gpt-4o-mini` (configurable)
- **Embedding Model**: OpenAI `text-embedding-3-small` (configurable)
- **Document Retrieval**: Top 5 documents minimum for context
- **Reranking**: Enabled by default for better document selection