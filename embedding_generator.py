import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import logging
import os
import pickle
import time
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel

class EmbeddingGenerator:
    """
    Generates embeddings for code snippets using pre-trained models with advanced features.
    """
    
    def __init__(self, config):
        """
        Initialize the EmbeddingGenerator with configuration.
        
        Args:
            config: Configuration object containing embedding settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_name = config.embedding_model
        self.cache_dir = os.path.join(config.base_dir, "cache", "embeddings")
        self.batch_size = config.batch_size
        self.use_cache = config.use_embedding_cache
        self.max_workers = config.max_workers
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load the model
        self.logger.info(f"Loading embedding model: {self.model_name}")
        try:
            # Try to load as SentenceTransformer first
            self.model = SentenceTransformer(self.model_name)
            self.model_type = "sentence_transformer"
            
            # Check if GPU is available and move model to GPU if possible
            if torch.cuda.is_available() and config.use_gpu:
                self.model = self.model.to(torch.device("cuda"))
                self.logger.info("Using GPU for embedding generation")
            else:
                self.logger.info("Using CPU for embedding generation")
                
        except Exception as e:
            self.logger.warning(f"Failed to load as SentenceTransformer: {str(e)}")
            try:
                # Try to load as Hugging Face model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model_type = "huggingface"
                
                # Move to GPU if available
                if torch.cuda.is_available() and config.use_gpu:
                    self.model = self.model.to(torch.device("cuda"))
                    self.logger.info("Using GPU for embedding generation with Hugging Face model")
                else:
                    self.logger.info("Using CPU for embedding generation with Hugging Face model")
            except Exception as e2:
                self.logger.error(f"Error loading embedding model: {str(e2)}")
                raise RuntimeError(f"Failed to load embedding model: {str(e2)}")
        
        # Initialize cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_embeddings(self, parsed_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all code chunks in parsed files with progress tracking.
        
        Args:
            parsed_files: List of dictionaries containing parsed file information
            
        Returns:
            List of dictionaries containing code information with embeddings
        """
        self.logger.info(f"Generating embeddings for {len(parsed_files)} files")
        all_embeddings_data = []
        all_chunks = []
        chunk_metadata = []
        
        # Collect all chunks from all files
        for file_data in parsed_files:
            if file_data is None:
                continue
                
            file_path = file_data['file_path']
            language = file_data['language']
            
            for chunk_idx, chunk in enumerate(file_data['chunks']):
                code = chunk['code']
                line_numbers = chunk['line_numbers']
                
                all_chunks.append((code, language))
                chunk_metadata.append({
                    'file_path': file_path,
                    'language': language,
                    'code': code,
                    'line_numbers': line_numbers
                })
        
        # Generate embeddings in batches with progress tracking
        total_chunks = len(all_chunks)
        self.logger.info(f"Processing {total_chunks} code chunks")
        
        if total_chunks == 0:
            return []
        
        # Process in batches
        embeddings = []
        
        if self.max_workers > 1 and total_chunks > self.batch_size:
            # Use parallel processing for large datasets
            embeddings = self._generate_embeddings_parallel(all_chunks)
        else:
            # Use batch processing for smaller datasets
            embeddings = self._generate_embeddings_batched(all_chunks)
        
        # Combine embeddings with metadata
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                metadata = chunk_metadata[i].copy()
                metadata['embedding'] = embedding
                all_embeddings_data.append(metadata)
        
        # Log cache statistics if using cache
        if self.use_cache:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
            self.logger.info(f"Cache statistics: {self.cache_hits} hits, {self.cache_misses} misses, {hit_rate:.2f}% hit rate")
        
        self.logger.info(f"Generated {len(all_embeddings_data)} embeddings")
        return all_embeddings_data
    
    def _generate_embeddings_batched(self, chunks: List[Tuple[str, str]]) -> List[np.ndarray]:
        """
        Generate embeddings in batches.
        
        Args:
            chunks: List of (code, language) tuples
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = self.batch_size
        
        # Process in batches with progress tracking
        with tqdm(total=len(chunks), desc="Generating embeddings", unit="chunk") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_codes = [self.preprocess_code(code, lang) for code, lang in batch_chunks]
                
                # Check cache for each chunk in the batch
                batch_embeddings = []
                uncached_indices = []
                uncached_codes = []
                
                if self.use_cache:
                    for j, code in enumerate(batch_codes):
                        cache_key = self._get_cache_key(code)
                        cached_embedding = self._get_from_cache(cache_key)
                        
                        if cached_embedding is not None:
                            batch_embeddings.append(cached_embedding)
                            self.cache_hits += 1
                        else:
                            batch_embeddings.append(None)
                            uncached_indices.append(j)
                            uncached_codes.append(code)
                            self.cache_misses += 1
                else:
                    uncached_indices = list(range(len(batch_codes)))
                    uncached_codes = batch_codes
                    batch_embeddings = [None] * len(batch_codes)
                
                # Generate embeddings for uncached codes
                if uncached_codes:
                    if self.model_type == "sentence_transformer":
                        new_embeddings = self.model.encode(uncached_codes, show_progress_bar=False)
                    else:  # huggingface
                        new_embeddings = self._generate_huggingface_embeddings(uncached_codes)
                    
                    # Update batch embeddings and cache
                    for k, idx in enumerate(uncached_indices):
                        embedding = new_embeddings[k]
                        batch_embeddings[idx] = embedding
                        
                        if self.use_cache:
                            cache_key = self._get_cache_key(batch_codes[idx])
                            self._save_to_cache(cache_key, embedding)
                
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch_chunks))
        
        return embeddings
    
    def _generate_embeddings_parallel(self, chunks: List[Tuple[str, str]]) -> List[np.ndarray]:
        """
        Generate embeddings using parallel processing.
        
        Args:
            chunks: List of (code, language) tuples
            
        Returns:
            List of embedding vectors
        """
        # Preprocess all codes first
        preprocessed_chunks = [(self.preprocess_code(code, lang), i) for i, (code, lang) in enumerate(chunks)]
        
        # Prepare result container
        embeddings = [None] * len(chunks)
        
        # Process chunks in parallel
        with tqdm(total=len(chunks), desc="Generating embeddings", unit="chunk") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks
                future_to_idx = {}
                for preprocessed_code, idx in preprocessed_chunks:
                    if self.use_cache:
                        cache_key = self._get_cache_key(preprocessed_code)
                        cached_embedding = self._get_from_cache(cache_key)
                        
                        if cached_embedding is not None:
                            embeddings[idx] = cached_embedding
                            self.cache_hits += 1
                            pbar.update(1)
                            continue
                        
                        self.cache_misses += 1
                    
                    future = executor.submit(self._generate_single_embedding, preprocessed_code)
                    future_to_idx[future] = (idx, preprocessed_code)
                
                # Process completed tasks
                for future in as_completed(future_to_idx):
                    idx, preprocessed_code = future_to_idx[future]
                    try:
                        embedding = future.result()
                        embeddings[idx] = embedding
                        
                        if self.use_cache:
                            cache_key = self._get_cache_key(preprocessed_code)
                            self._save_to_cache(cache_key, embedding)
                    except Exception as e:
                        self.logger.error(f"Error generating embedding: {str(e)}")
                    
                    pbar.update(1)
        
        return embeddings
    
    def _generate_single_embedding(self, code: str) -> np.ndarray:
        """
        Generate embedding for a single preprocessed code snippet.
        
        Args:
            code: Preprocessed code snippet
            
        Returns:
            Embedding vector
        """
        if self.model_type == "sentence_transformer":
            return self.model.encode(code, show_progress_bar=False)
        else:  # huggingface
            return self._generate_huggingface_embeddings([code])[0]
    
    def _generate_huggingface_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Hugging Face models.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embedding vectors
        """
        # Tokenize and prepare inputs
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Move to GPU if available
        if torch.cuda.is_available() and self.config.use_gpu:
            encoded_input = {k: v.to(torch.device("cuda")) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Use mean of last hidden states as embeddings
        embeddings = model_output.last_hidden_state.mean(dim=1)
        
        # Move back to CPU and convert to numpy
        return embeddings.cpu().numpy()
    
    def _get_cache_key(self, code: str) -> str:
        """
        Generate a cache key for a code snippet.
        
        Args:
            code: Code snippet
            
        Returns:
            Cache key string
        """
        return hashlib.md5(code.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached embedding or None if not found
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading from cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> bool:
        """
        Save embedding to cache.
        
        Args:
            cache_key: Cache key
            embedding: Embedding vector
            
        Returns:
            Boolean indicating success
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            return True
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False
    
    def generate_embedding(self, code: str, language: str) -> np.ndarray:
        """
        Generate embedding for a single code chunk.
        
        Args:
            code: Code snippet
            language: Programming language of the code
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Preprocess code for better embedding quality
        preprocessed_code = self.preprocess_code(code, language)
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(preprocessed_code)
            cached_embedding = self._get_from_cache(cache_key)
            
            if cached_embedding is not None:
                self.cache_hits += 1
                return cached_embedding
            
            self.cache_misses += 1
        
        # Generate embedding
        if self.model_type == "sentence_transformer":
            embedding = self.model.encode(preprocessed_code, show_progress_bar=False)
        else:  # huggingface
            embedding = self._generate_huggingface_embeddings([preprocessed_code])[0]
        
        # Save to cache if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(preprocessed_code)
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def preprocess_code(self, code: str, language: str) -> str:
        """
        Preprocess code to improve embedding quality with advanced language-specific handling.
        
        Args:
            code: Code snippet
            language: Programming language of the code
            
        Returns:
            Preprocessed code string
        """
        # Handle empty code
        if not code or code.isspace():
            return ""
        
        # Normalize language name
        language = language.lower()
        
        # Apply language-specific preprocessing
        if language in ['python', 'py']:
            processed_code = self._preprocess_python(code)
        elif language in ['javascript', 'js', 'typescript', 'ts']:
            processed_code = self._preprocess_javascript(code)
        elif language in ['java']:
            processed_code = self._preprocess_java(code)
        elif language in ['c', 'cpp', 'c++', 'csharp', 'c#']:
            processed_code = self._preprocess_c_family(code)
        else:
            # Generic preprocessing for other languages
            processed_code = self._preprocess_generic(code)
        
        # Add language context if configured
        if self.config.add_language_context:
            processed_code = f"[{language}] {processed_code}"
        
        return processed_code
    
    def _preprocess_python(self, code: str) -> str:
        """
        Preprocess Python code.
        
        Args:
            code: Python code
            
        Returns:
            Preprocessed code
        """
        import re
        
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Normalize whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(line for line in lines if line.strip())
        
        # Normalize string literals
        code = re.sub(r'"[^"]*"', '"STR"', code)
        code = re.sub(r"'[^']*'", "'STR'", code)
        
        # Normalize numbers
        code = re.sub(r'\b\d+\b', 'NUM', code)
        
        return code
    
    def _preprocess_javascript(self, code: str) -> str:
        """
        Preprocess JavaScript/TypeScript code.
        
        Args:
            code: JS/TS code
            
        Returns:
            Preprocessed code
        """
        import re
        
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(line for line in lines if line.strip())
        
        # Normalize string literals
        code = re.sub(r'"[^"]*"', '"STR"', code)
        code = re.sub(r"'[^']*'", "'STR'", code)
        code = re.sub(r'`[^`]*`', '`STR`', code)
        
        # Normalize numbers
        code = re.sub(r'\b\d+\b', 'NUM', code)
        
        # Handle JSX/TSX
        code = re.sub(r'<[^>]*>', '<JSX>', code)
        
        return code
    
    def _preprocess_java(self, code: str) -> str:
        """
        Preprocess Java code.
        
        Args:
            code: Java code
            
        Returns:
            Preprocessed code
        """
        import re
        
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove Javadoc comments
        code = re.sub(r'/\*\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(line for line in lines if line.strip())
        
        # Normalize string literals
        code = re.sub(r'"[^"]*"', '"STR"', code)
        
        # Normalize numbers
        code = re.sub(r'\b\d+\b', 'NUM', code)
        
        # Extract class and method names for better context
        class_matches = re.findall(r'class\s+(\w+)', code)
        method_matches = re.findall(r'(?:public|private|protected)?\s+(?:static)?\s+\w+\s+(\w+)\s*\(', code)
        
        class_context = f"Classes: {', '.join(class_matches)} " if class_matches else ""
        method_context = f"Methods: {', '.join(method_matches)} " if method_matches else ""
        
        # Add extracted context at the beginning
        if class_context or method_context:
            code = f"{class_context}{method_context}\n{code}"
        
        return code
    
    def _preprocess_c_family(self, code: str) -> str:
        """
        Preprocess C-family code (C, C++, C#).
        
        Args:
            code: C-family code
            
        Returns:
            Preprocessed code
        """
        import re
        
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(line for line in lines if line.strip())
        
        # Normalize string literals
        code = re.sub(r'"[^"]*"', '"STR"', code)
        
        # Normalize numbers
        code = re.sub(r'\b\d+\b', 'NUM', code)
        
        # Normalize preprocessor directives
        code = re.sub(r'#include\s*<[^>]*>', '#include <HEADER>', code)
        code = re.sub(r'#include\s*"[^"]*"', '#include "HEADER"', code)
        code = re.sub(r'#define\s+\w+\s+.*$', '#define MACRO', code, flags=re.MULTILINE)
        
        return code
    
    def _preprocess_generic(self, code: str) -> str:
        """
        Generic preprocessing for any code.
        
        Args:
            code: Code in any language
            
        Returns:
            Preprocessed code
        """
        # Remove excessive whitespace but preserve some structure
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(line for line in lines if line.strip())
        
        return code
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess search query to improve search quality.
        
        Args:
            query: Search query string
            
        Returns:
            Preprocessed query string
        """
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Add search context if configured
        if self.config.add_search_context:
            query = f"Find code that: {query}"
        
        return query
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array containing the query embedding vector
        """
        # Preprocess query
        preprocessed_query = self.preprocess_query(query)
        
        # Generate embedding
        if self.model_type == "sentence_transformer":
            embedding = self.model.encode(preprocessed_query, show_progress_bar=False)
        else:  # huggingface
            embedding = self._generate_huggingface_embeddings([preprocessed_query])[0]
        
        return embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
        
        # Count cache files
        cache_files = 0
        cache_size = 0
        
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    cache_files += 1
                    cache_size += os.path.getsize(os.path.join(self.cache_dir, file))
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_files': cache_files,
            'cache_size_bytes': cache_size,
            'cache_size_mb': cache_size / (1024 * 1024) if cache_size > 0 else 0
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the embedding cache.
        
        Returns:
            Boolean indicating success
        """
        if not os.path.exists(self.cache_dir):
            return True
        
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
            
            self.cache_hits = 0
            self.cache_misses = 0
            
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False

