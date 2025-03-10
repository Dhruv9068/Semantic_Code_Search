import numpy as np
import os
import pickle
import logging
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEmbeddings:
    """
    Generates and manages embeddings for code elements.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embeddings_dir: str = "data/embeddings"):
        self.model_name = model_name
        self.embeddings_dir = embeddings_dir
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        logger.info(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded")
    
    def generate_embeddings(self, code_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for code elements.
        
        Args:
            code_elements: List of dictionaries containing code elements
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        logger.info(f"Generating embeddings for {len(code_elements)} code elements...")
        
        # Prepare texts for embedding
        texts = []
        for element in code_elements:
            # Combine name, docstring, and code for better semantic understanding
            text = f"{element['name']} {element['docstring']} {element['code']}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create embeddings dictionary
        embeddings_dict = {
            "model_name": self.model_name,
            "elements": code_elements,
            "embeddings": embeddings
        }
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return embeddings_dict
    
    def save_embeddings(self, embeddings_dict: Dict[str, Any], name: str) -> str:
        """
        Save embeddings to disk.
        
        Args:
            embeddings_dict: Dictionary containing embeddings and metadata
            name: Name of the embeddings file
            
        Returns:
            Path to the saved embeddings file
        """
        file_path = os.path.join(self.embeddings_dir, f"{name}.pkl")
        
        logger.info(f"Saving embeddings to {file_path}...")
        
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        logger.info(f"Embeddings saved to {file_path}")
        
        return file_path
    
    def load_embeddings(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load embeddings from disk.
        
        Args:
            name: Name of the embeddings file
            
        Returns:
            Dictionary containing embeddings and metadata or None if file doesn't exist
        """
        file_path = os.path.join(self.embeddings_dir, f"{name}.pkl")
        
        if not os.path.exists(file_path):
            logger.error(f"Embeddings file {file_path} does not exist")
            return None
        
        logger.info(f"Loading embeddings from {file_path}...")
        
        with open(file_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        logger.info(f"Loaded embeddings for {len(embeddings_dict['elements'])} code elements")
        
        return embeddings_dict
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector for the query
        """
        return self.model.encode(query)

