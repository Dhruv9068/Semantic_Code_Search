import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manages the search index for code elements.
    """
    
    def __init__(self, index_dir: str = "data/indexes"):
        self.index_dir = index_dir
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
    
    def create_index(self, embeddings_dict: Dict[str, Any], name: str) -> str:
        """
        Create a search index from embeddings.
        
        Args:
            embeddings_dict: Dictionary containing embeddings and metadata
            name: Name of the index
            
        Returns:
            Path to the saved index file
        """
        logger.info(f"Creating index {name}...")
        
        # The index is just the embeddings dictionary for now
        # In a more advanced implementation, we could use a more efficient index structure
        index = embeddings_dict
        
        file_path = os.path.join(self.index_dir, f"{name}.idx")
        
        with open(file_path, 'wb') as f:
            pickle.dump(index, f)
        
        logger.info(f"Index created and saved to {file_path}")
        
        return file_path
    
    def load_index(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a search index from disk.
        
        Args:
            name: Name of the index
            
        Returns:
            Dictionary containing the index or None if file doesn't exist
        """
        file_path = os.path.join(self.index_dir, f"{name}.idx")
        
        if not os.path.exists(file_path):
            logger.error(f"Index file {file_path} does not exist")
            return None
        
        logger.info(f"Loading index from {file_path}...")
        
        with open(file_path, 'rb') as f:
            index = pickle.load(f)
        
        logger.info(f"Loaded index with {len(index['elements'])} code elements")
        
        return index
    
    def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        indexes = []
        
        for file in os.listdir(self.index_dir):
            if file.endswith('.idx'):
                indexes.append(file[:-4])  # Remove .idx extension
        
        return indexes
    
    def delete_index(self, name: str) -> bool:
        """
        Delete an index.
        
        Args:
            name: Name of the index
            
        Returns:
            True if index was deleted, False otherwise
        """
        file_path = os.path.join(self.index_dir, f"{name}.idx")
        
        if not os.path.exists(file_path):
            logger.error(f"Index file {file_path} does not exist")
            return False
        
        logger.info(f"Deleting index {file_path}...")
        
        os.remove(file_path)
        
        logger.info(f"Index {name} deleted")
        
        return True

