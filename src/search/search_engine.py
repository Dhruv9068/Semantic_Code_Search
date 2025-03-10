import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from ..indexer.embeddings import CodeEmbeddings
from ..indexer.index_manager import IndexManager
from .query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
   """
   Core search functionality for code search.
   """
   
   def __init__(self):
       self.embeddings = CodeEmbeddings()
       self.index_manager = IndexManager()
       self.query_processor = QueryProcessor()
       self.current_index = None
       self.current_index_name = None
   
   def load_index(self, index_name: str) -> bool:
       """
       Load a search index.
       
       Args:
           index_name: Name of the index
           
       Returns:
           True if index was loaded successfully, False otherwise
       """
       index = self.index_manager.load_index(index_name)
       
       if index is None:
           return False
       
       self.current_index = index
       self.current_index_name = index_name
       
       return True
   
   def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
       """
       Search for code elements matching a query.
       
       Args:
           query: Natural language query
           top_k: Number of results to return
           
       Returns:
           List of dictionaries containing search results
       """
       if self.current_index is None:
           logger.error("No index loaded")
           return []
       
       # Process query and extract filters
       processed_query, filters = self.query_processor.extract_filters(query)
       
       # Generate query embedding
       query_embedding = self.embeddings.generate_query_embedding(processed_query)
       
       # Get embeddings and elements from the index
       index_embeddings = self.current_index["embeddings"]
       elements = self.current_index["elements"]
       
       # Calculate cosine similarity
       similarities = self._cosine_similarity(query_embedding, index_embeddings)
       
       # Sort by similarity
       sorted_indices = np.argsort(similarities)[::-1]
       
       # Apply filters but don't limit results by top_k yet
       results = []
       for idx in sorted_indices:
           element = elements[idx]
           
           # Apply type filter if present
           if "type" in filters and element["type"] != filters["type"]:
               continue
           
           # Add similarity score to the element
           element_with_score = element.copy()
           element_with_score["score"] = float(similarities[idx])
           
           results.append(element_with_score)
       
       # Now limit to top_k after collecting all matching results
       results = results[:top_k]
       
       logger.info(f"Found {len(results)} results for query: {query}")
       
       return results
   
   def _cosine_similarity(self, query_embedding: np.ndarray, index_embeddings: np.ndarray) -> np.ndarray:
       """
       Calculate cosine similarity between query embedding and index embeddings.
       
       Args:
           query_embedding: Query embedding vector
           index_embeddings: Index embedding matrix
           
       Returns:
           Array of similarity scores
       """
       # Normalize query embedding
       query_norm = np.linalg.norm(query_embedding)
       if query_norm > 0:
           query_embedding = query_embedding / query_norm
       
       # Normalize index embeddings
       index_norms = np.linalg.norm(index_embeddings, axis=1, keepdims=True)
       normalized_index_embeddings = np.divide(index_embeddings, index_norms, 
                                             out=np.zeros_like(index_embeddings), 
                                             where=index_norms > 0)
       
       # Calculate cosine similarity
       similarities = np.dot(normalized_index_embeddings, query_embedding)
       
       return similarities
   
   def get_available_indexes(self) -> List[str]:
       """
       Get a list of available indexes.
       
       Returns:
           List of index names
       """
       return self.index_manager.list_indexes()
   
   def get_current_index_name(self) -> Optional[str]:
       """
       Get the name of the currently loaded index.
       
       Returns:
           Name of the current index or None if no index is loaded
       """
       return self.current_index_name

