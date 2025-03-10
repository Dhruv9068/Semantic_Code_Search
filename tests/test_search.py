import unittest
import os
import tempfile
import shutil
import numpy as np
from src.search.search_engine import SearchEngine
from src.indexer.embeddings import CodeEmbeddings
from src.indexer.index_manager import IndexManager

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create temporary directories for embeddings and indexes
        self.embeddings_dir = os.path.join(self.temp_dir, 'embeddings')
        self.index_dir = os.path.join(self.temp_dir, 'indexes')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Create test embeddings and index
        self.embeddings = CodeEmbeddings(embeddings_dir=self.embeddings_dir)
        self.index_manager = IndexManager(index_dir=self.index_dir)
        self.search_engine = SearchEngine()
        
        # Override the directories in the search engine
        self.search_engine.embeddings = self.embeddings
        self.search_engine.index_manager = self.index_manager
        
        # Create a test index
        self.create_test_index()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_test_index(self):
        # Create test code elements
        code_elements = [
            {
                'type': 'function',
                'name': 'parse_json',
                'docstring': 'Parse a JSON string into a Python object.',
                'code': 'def parse_json(json_str):\n    import json\n    return json.loads(json_str)',
                'file_path': '/path/to/file.py',
                'line_number': 1,
                'args': ['json_str'],
                'returns': None
            },
            {
                'type': 'function',
                'name': 'format_json',
                'docstring': 'Format a Python object as a JSON string.',
                'code': 'def format_json(obj):\n    import json\n    return json.dumps(obj, indent=2)',
                'file_path': '/path/to/file.py',
                'line_number': 5,
                'args': ['obj'],
                'returns': None
            },
            {
                'type': 'class',
                'name': 'Calculator',
                'docstring': 'A simple calculator class.',
                'code': 'class Calculator:\n    def add(self, a, b):\n        return a + b',
                'file_path': '/path/to/file.py',
                'line_number': 10,
                'bases': []
            }
        ]
        
        # Create fake embeddings (random vectors)
        embeddings = np.random.rand(len(code_elements), 384)  # 384 is the dimension of the all-MiniLM-L6-v2 model
        
        # Create embeddings dictionary
        embeddings_dict = {
            'model_name': 'test-model',
            'elements': code_elements,
            'embeddings': embeddings
        }
        
        # Save embeddings
        self.embeddings.save_embeddings(embeddings_dict, 'test-index')
        
        # Create index
        self.index_manager.create_index(embeddings_dict, 'test-index')
    
    def test_load_index(self):
        # Test loading the index
        success = self.search_engine.load_index('test-index')
        self.assertTrue(success)
        self.assertEqual(self.search_engine.current_index_name, 'test-index')
    
    def test_search(self):
        # Load the index
        self.search_engine.load_index('test-index')
        
        # Mock the generate_query_embedding method to return a specific vector
        # that will match the first element in our test index
        original_method = self.search_engine.embeddings.generate_query_embedding
        
        def mock_generate_query_embedding(query):
            # Return a vector that will have high similarity with the first element
            return self.search_engine.current_index['embeddings'][0]
        
        self.search_engine.embeddings.generate_query_embedding = mock_generate_query_embedding
        
        # Search for JSON parsing
        results = self.search_engine.search('parse json')
        
        # Restore the original method
        self.search_engine.embeddings.generate_query_embedding = original_method
        
        # Check that we got results
        self.assertGreater(len(results), 0)
        
        # The first result should be the parse_json function
        self.assertEqual(results[0]['name'], 'parse_json')

if __name__ == '__main__':
    unittest.main()

