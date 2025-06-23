

1. Basic Usage
    a.  Interactive mode - asks for questions one at a time
            python main.py
    b.  Ask a specific question
            python main.py --questions "What is RAG?"
    c.  Ask multiple questions
            python main.py --questions "What is RAG?" "How does vector search work?"
    d.  Process questions from a file (one per line)
            python main.py --questions_file my_questions.txt


2. Using Different LLM Models
    a.  Use a specific OpenAI model
            python main.py --llm_model gpt-4o
    b.  Use an Ollama model
            python main.py --llm_provider ollama --llm_model llama2


3. Using Different Embedding Models
    a.  Use different OpenAI embedding model
            python main.py --embedding_model text-embedding-3-large
    b.  Use Sentence Transformer embedding model
            python main.py --embedding_type sentence_transformer --embedding_model all-mpnet-base-v2


4. Configuring Pinecone
    a.  Specify a different index
            python main.py --pinecone_index my-knowledge-base
    b.  Use a specific namespace within an index
            python main.py --pinecone_index my-knowledge-base --pinecone_namespace corporate-docs


5. Using a Configuration File
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


6. Common Examples
    a.  Ask a specific question against a particular namespace
            python main.py --questions "What are our Q2 revenue projections?" --pinecone_namespace financial-reports
    b.  Process multiple questions with a specific model
            python main.py --questions_file quarterly_questions.txt --llm_model gpt-4o
    c.  Use a specific embedding model to match what was used during document ingestion
            python main.py --embedding_model text-embedding-3-small


7. Advanced Options
    a.  Turn off reranking for faster but potentially less accurate responses
            python main.py --config config.json --no_reranking
    b.  Save outputs for later reference
            python main.py --questions_file important_questions.txt --save_outputs
    c.  Compare to ground truth answers
            python main.py --questions_file test_questions.txt --ground_truth ground_truth.json