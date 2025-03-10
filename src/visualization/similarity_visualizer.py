import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityVisualizer:
    """
    Visualizes code similarity and relationships.
    """
    
    def __init__(self):
        pass
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray, 
                                   labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Visualize a similarity matrix as a heatmap.
        
        Args:
            similarity_matrix: Similarity matrix
            labels: Optional labels for the matrix
            
        Returns:
            Dictionary containing the visualization image
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Similarity')
            
            if labels is not None:
                # If there are too many labels, show only a subset
                if len(labels) > 20:
                    tick_indices = np.linspace(0, len(labels) - 1, 20, dtype=int)
                    plt.xticks(tick_indices, [labels[i] for i in tick_indices], rotation=90)
                    plt.yticks(tick_indices, [labels[i] for i in tick_indices])
                else:
                    plt.xticks(range(len(labels)), labels, rotation=90)
                    plt.yticks(range(len(labels)), labels)
            
            plt.title('Code Similarity Matrix')
            plt.tight_layout()
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'image': f"data:image/png;base64,{image_base64}"
            }
            
        except Exception as e:
            logger.error(f"Error visualizing similarity matrix: {str(e)}")
            return {
                'error': str(e)
            }
    
    def visualize_embeddings_2d(self, embeddings: np.ndarray, 
                              labels: Optional[List[str]] = None,
                              types: Optional[List[str]] = None,
                              method: str = 'tsne') -> Dict[str, Any]:
        """
        Visualize code embeddings in 2D.
        
        Args:
            embeddings: Code embeddings
            labels: Optional labels for the embeddings
            types: Optional types for the embeddings
            method: Dimensionality reduction method ('tsne' or 'pca')
            
        Returns:
            Dictionary containing the visualization image
        """
        try:
            # Reduce dimensionality to 2D
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
            else:  # pca
                reducer = PCA(n_components=2, random_state=42)
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 10))
            
            # If types are provided, use different colors for each type
            if types is not None:
                unique_types = list(set(types))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
                
                for i, type_name in enumerate(unique_types):
                    indices = [j for j, t in enumerate(types) if t == type_name]
                    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                              c=[colors[i]], label=type_name, alpha=0.7)
                
                plt.legend()
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
            
            # If labels are provided and there aren't too many, add them to the plot
            if labels is not None and len(labels) <= 50:
                for i, label in enumerate(labels):
                    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                               fontsize=8, alpha=0.7)
            
            plt.title(f'Code Embeddings Visualization ({method.upper()})')
            plt.tight_layout()
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'image': f"data:image/png;base64,{image_base64}",
                'embeddings_2d': embeddings_2d.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {str(e)}")
            return {
                'error': str(e)
            }
    
    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                         labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Visualize code clusters.
        
        Args:
            embeddings: Code embeddings
            cluster_labels: Cluster labels for each embedding
            labels: Optional labels for the embeddings
            
        Returns:
            Dictionary containing the visualization image
        """
        try:
            # Reduce dimensionality to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 10))
            
            # Get unique clusters
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            # Plot each cluster
            for i, cluster in enumerate(unique_clusters):
                indices = np.where(cluster_labels == cluster)[0]
                plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                          c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
            
            plt.legend()
            
            # If labels are provided and there aren't too many, add them to the plot
            if labels is not None and len(labels) <= 50:
                for i, label in enumerate(labels):
                    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                               fontsize=8, alpha=0.7)
            
            plt.title('Code Clusters Visualization')
            plt.tight_layout()
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'image': f"data:image/png;base64,{image_base64}",
                'embeddings_2d': embeddings_2d.tolist(),
                'cluster_labels': cluster_labels.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error visualizing clusters: {str(e)}")
            return {
                'error': str(e)
            }

