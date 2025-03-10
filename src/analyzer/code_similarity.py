import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeSimilarity:
    """
    Analyzes similarity between code snippets using their embeddings.
    """
    
    def __init__(self):
        pass
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute a similarity matrix for a set of code embeddings.
        
        Args:
            embeddings: Matrix of code embeddings (n_samples, n_features)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(embeddings, norms, 
                                        out=np.zeros_like(embeddings), 
                                        where=norms > 0)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(normalized_embeddings)
        
        return similarity_matrix
    
    def find_similar_code(self, query_embedding: np.ndarray, code_embeddings: np.ndarray, 
                         threshold: float = 0.7, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find code snippets similar to a query.
        
        Args:
            query_embedding: Embedding of the query code
            code_embeddings: Matrix of code embeddings
            threshold: Similarity threshold
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Normalize code embeddings
        norms = np.linalg.norm(code_embeddings, axis=1, keepdims=True)
        normalized_code_embeddings = np.divide(code_embeddings, norms, 
                                             out=np.zeros_like(code_embeddings), 
                                             where=norms > 0)
        
        # Compute similarities
        similarities = np.dot(normalized_code_embeddings, query_embedding)
        
        # Find indices of similar code snippets
        similar_indices = np.where(similarities >= threshold)[0]
        
        # Sort by similarity (descending)
        similar_indices = similar_indices[np.argsort(-similarities[similar_indices])]
        
        # Limit to top_k results
        similar_indices = similar_indices[:top_k]
        
        # Create result list
        results = [(int(idx), float(similarities[idx])) for idx in similar_indices]
        
        return results
    
    def find_duplicate_code(self, embeddings: np.ndarray, threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """
        Find potential duplicate code snippets.
        
        Args:
            embeddings: Matrix of code embeddings
            threshold: Similarity threshold
            
        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Find pairs of similar code snippets
        # We only consider the upper triangle of the matrix to avoid duplicates
        n_samples = similarity_matrix.shape[0]
        duplicates = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    duplicates.append((i, j, float(similarity)))
        
        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        return duplicates
    
    def compute_code_clusters(self, embeddings: np.ndarray, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster code snippets based on their embeddings.
        
        Args:
            embeddings: Matrix of code embeddings
            n_clusters: Number of clusters
            
        Returns:
            Dictionary containing cluster information
        """
        try:
            from sklearn.cluster import KMeans
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Compute distance to cluster centers
            distances = np.sqrt(((embeddings - kmeans.cluster_centers_[cluster_labels]) ** 2).sum(axis=1))
            
            # Compute cluster sizes
            cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
            
            # Compute average distance within clusters
            cluster_distances = np.zeros(n_clusters)
            for i in range(n_clusters):
                if cluster_sizes[i] > 0:
                    cluster_distances[i] = distances[cluster_labels == i].mean()
            
            return {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'cluster_sizes': cluster_sizes,
                'cluster_distances': cluster_distances,
                'inertia': kmeans.inertia_
            }
            
        except ImportError:
            logger.error("scikit-learn is required for clustering")
            return {
                'error': 'scikit-learn is required for clustering'
            }
    
    def compute_similarity_to_reference(self, reference_embedding: np.ndarray, 
                                      code_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity of code snippets to a reference snippet.
        
        Args:
            reference_embedding: Embedding of the reference code
            code_embeddings: Matrix of code embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize reference embedding
        reference_norm = np.linalg.norm(reference_embedding)
        if reference_norm > 0:
            reference_embedding = reference_embedding / reference_norm
        
        # Normalize code embeddings
        norms = np.linalg.norm(code_embeddings, axis=1, keepdims=True)
        normalized_code_embeddings = np.divide(code_embeddings, norms, 
                                             out=np.zeros_like(code_embeddings), 
                                             where=norms > 0)
        
        # Compute similarities
        similarities = np.dot(normalized_code_embeddings, reference_embedding)
        
        return similarities

