import numpy as np
from typing import List, Dict, Any
import logging
import faiss

class Searcher:
    """
    Searches indexed code using semantic similarity.
    """
    
    def __init__(self, config):
        """
        Initialize the Searcher with configuration.
        
        Args:
            config: Configuration object containing searcher settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indexer = None
        self.embedding_generator = None
    
    def set_components(self, indexer, embedding_generator):
        """
        Set required components.
        
        Args:
            indexer: Indexer instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.indexer = indexer
        self.embedding_generator = embedding_generator
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code snippets similar to the query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        self.logger.info(f"Searching for: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Reshape for FAISS
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in the index
        k = min(num_results * 2, self.indexer.metadata['num_documents'])  # Fetch more results to filter duplicates
        if k == 0:
            return []
        
        distances, indices = self.indexer.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        seen_files = set()
        for i, idx in enumerate(indices[0]):
            if idx < len(self.indexer.code_data):
                code_info = self.indexer.code_data[idx]
                
                # Skip if we've already included this file
                if code_info['file_path'] in seen_files:
                    continue
                
                # Calculate similarity score (convert distance to similarity)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                result = {
                    'file_path': code_info['file_path'],
                    'language': code_info['language'],
                    'code': code_info['code'],
                    'line_numbers': code_info['line_numbers'],
                    'similarity': similarity
                }
                
                results.append(result)
                seen_files.add(code_info['file_path'])
                
                if len(results) >= num_results:
                    break
        
        return results
    
    def filter_results(self, results: List[Dict[str, Any]], 
                       language: str = None) -> List[Dict[str, Any]]:
        """
        Filter search results based on criteria.
        
        Args:
            results: List of search results
            language: Filter by programming language
            
        Returns:
            Filtered list of search results
        """
        filtered_results = results
        
        # Filter by language
        if language:
            filtered_results = [r for r in filtered_results if r['language'].lower() == language.lower()]
        
        return filtered_results

