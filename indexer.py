import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any
import logging
import pickle
from datetime import datetime

class Indexer:
    """
    Indexes code embeddings for efficient similarity search.
    """
    
    def __init__(self, config):
        """
        Initialize the Indexer with configuration.
        
        Args:
            config: Configuration object containing indexer settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index_dir = config.index_dir
        self.metadata_file = os.path.join(self.index_dir, 'metadata.json')
        self.index_file = os.path.join(self.index_dir, 'faiss_index.bin')
        self.data_file = os.path.join(self.index_dir, 'code_data.pkl')
        
        # Create index directory if it doesn't exist
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        # Load existing index if available
        self.load_index()
    
    def initialize_index(self):
        """
        Initialize an empty index.
        """
        self.logger.info("Initializing empty index")
        
        # Initialize empty metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'num_documents': 0,
            'embedding_dimension': self.config.embedding_dimension,
            'languages': {},
            'repositories': {}
        }
        
        # Initialize empty code data
        self.code_data = []
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.config.embedding_dimension)
        
        # Save empty index
        self.save_index()
    
    def load_index(self):
        """
        Load existing index from disk.
        """
        try:
            # Check if index files exist
            if (os.path.exists(self.metadata_file) and 
                os.path.exists(self.index_file) and 
                os.path.exists(self.data_file)):
                
                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load code data
                with open(self.data_file, 'rb') as f:
                    self.code_data = pickle.load(f)
                
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                self.logger.info(f"Loaded index with {self.metadata['num_documents']} documents")
            else:
                self.logger.info("No existing index found, initializing empty index")
                self.initialize_index()
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            self.logger.info("Initializing new index")
            self.initialize_index()
    
    def save_index(self):
        """
        Save index to disk.
        """
        try:
            # Update metadata
            self.metadata['updated_at'] = datetime.now().isoformat()
            self.metadata['num_documents'] = len(self.code_data)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save code data
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.code_data, f)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            self.logger.info(f"Saved index with {self.metadata['num_documents']} documents")
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
    
    def index_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """
        Index code embeddings.
        
        Args:
            embeddings_data: List of dictionaries containing code information with embeddings
        """
        if not embeddings_data:
            self.logger.warning("No embeddings to index")
            return
        
        self.logger.info(f"Indexing {len(embeddings_data)} embeddings")
        
        # Extract embeddings as numpy array
        embeddings = np.array([data['embedding'] for data in embeddings_data], dtype=np.float32)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Update metadata
        for data in embeddings_data:
            # Update language statistics
            language = data['language']
            if language in self.metadata['languages']:
                self.metadata['languages'][language] += 1
            else:
                self.metadata['languages'][language] = 1
            
            # Update repository statistics
            repo_path = os.path.dirname(data['file_path'])
            if repo_path in self.metadata['repositories']:
                self.metadata['repositories'][repo_path] += 1
            else:
                self.metadata['repositories'][repo_path] = 1
            
            # Store code data without embedding (to save space)
            code_info = {
                'file_path': data['file_path'],
                'language': data['language'],
                'code': data['code'],
                'line_numbers': data['line_numbers']
            }
            self.code_data.append(code_info)
        
        # Save updated index
        self.save_index()
    
    def clear_index(self):
        """
        Clear the current index.
        """
        self.logger.info("Clearing index")
        self.initialize_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed code.
        
        Returns:
            Dictionary containing index statistics
        """
        return {
            'num_documents': self.metadata['num_documents'],
            'languages': self.metadata['languages'],
            'repositories': self.metadata['repositories'],
            'created_at': self.metadata['created_at'],
            'updated_at': self.metadata['updated_at']
        }

