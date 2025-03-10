import os
from typing import List, Dict, Any

class Config:
    """
    Enhanced configuration for the Semantic Code Search Engine.
    """
    
    def __init__(self):
        # Server configuration
        self.host = "127.0.0.1"
        self.port = 5000
        self.debug_mode = True
        
        # Directory configuration
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_dir = os.path.join(self.base_dir, "index")
        self.log_dir = os.path.join(self.base_dir, "logs")
        
        # Parser configuration
        self.supported_extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', 
            '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala',
            '.html', '.css', '.json', '.yaml', '.yml', '.md', '.sql'
        ]
        self.ignore_patterns = [
            r'node_modules', r'\.git', r'__pycache__', r'\.venv', 
            r'\.idea', r'\.vscode', r'\.DS_Store', r'\.ipynb_checkpoints',
            r'venv', r'env', r'dist', r'build', r'target', r'\.cache',
            r'\.next', r'\.nuxt', r'\.output', r'\.serverless', r'\.webpack'
        ]
        self.max_chunk_size = 100  # Maximum words per chunk
        self.min_chunk_size = 10   # Minimum words per chunk
        self.chunk_overlap = 5     # Number of lines to overlap between chunks
        
        # Embedding configuration
        self.embedding_model = "all-MiniLM-L6-v2"  # Sentence transformer model
        self.embedding_dimension = 384  # Dimension of embeddings from the model
        self.use_gpu = True  # Whether to use GPU for embedding generation
        self.add_language_context = True  # Add language as context to embeddings
        self.add_search_context = True  # Add search context to queries
        self.use_embedding_cache = True  # Use cache for embeddings
        self.batch_size = 32  # Batch size for embedding generation
        self.max_workers = 4  # Number of workers for parallel processing
        
        # Search configuration
        self.default_num_results = 10  # Default number of search results
        self.min_similarity_threshold = 0.5  # Minimum similarity score for results
        self.use_advanced_ranking = True  # Use advanced ranking algorithm
        self.rerank_results = True  # Rerank results based on additional criteria
        
        # UI configuration
        self.theme = "dark"  # UI theme (light or dark)
        self.highlight_theme = "monokai"  # Syntax highlighting theme
        self.max_code_height = 500  # Maximum height for code display in pixels
        self.enable_animations = True  # Enable UI animations
        
        # Advanced features
        self.enable_analytics = True  # Enable usage analytics
        self.enable_caching = True  # Enable caching for search results
        self.cache_ttl = 3600  # Cache time-to-live in seconds
        
        # Load environment-specific configuration
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Server configuration
        if os.environ.get('SEMANTIC_SEARCH_HOST'):
            self.host = os.environ.get('SEMANTIC_SEARCH_HOST')
        
        if os.environ.get('SEMANTIC_SEARCH_PORT'):
            self.port = int(os.environ.get('SEMANTIC_SEARCH_PORT'))
        
        if os.environ.get('SEMANTIC_SEARCH_DEBUG'):
            self.debug_mode = os.environ.get('SEMANTIC_SEARCH_DEBUG').lower() == 'true'
        
        # Embedding configuration
        if os.environ.get('SEMANTIC_SEARCH_MODEL'):
            self.embedding_model = os.environ.get('SEMANTIC_SEARCH_MODEL')
        
        if os.environ.get('SEMANTIC_SEARCH_USE_GPU'):
            self.use_gpu = os.environ.get('SEMANTIC_SEARCH_USE_GPU').lower() == 'true'
        
        # UI configuration
        if os.environ.get('SEMANTIC_SEARCH_THEME'):
            self.theme = os.environ.get('SEMANTIC_SEARCH_THEME')

