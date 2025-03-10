import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeGraph:
    """
    Generates visualizations of code relationships and structure.
    """
    
    def __init__(self):
        pass
    
    def generate_dependency_graph(self, code_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a dependency graph for code elements.
        
        Args:
            code_elements: List of code elements
            
        Returns:
            Dictionary containing graph data and image
        """
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes for each code element
            for i, element in enumerate(code_elements):
                G.add_node(i, label=element['name'], type=element['type'])
            
            # Add edges based on dependencies
            for i, element in enumerate(code_elements):
                # Check if this element references other elements
                for j, other_element in enumerate(code_elements):
                    if i != j and other_element['name'] in element['code']:
                        G.add_edge(i, j)
            
            # Generate graph image
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes with different colors based on type
            node_colors = {
                'function': 'skyblue',
                'class': 'lightgreen',
                'method': 'lightcoral',
                'module': 'lightyellow'
            }
            
            for node_type in node_colors:
                nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == node_type]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors[node_type], 
                                     node_size=500, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Draw labels
            labels = {n: attr['label'] for n, attr in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'graph': G,
                'image': f"data:image/png;base64,{image_base64}",
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error generating dependency graph: {str(e)}")
            return {
                'error': str(e)
            }
    
    def generate_call_graph(self, code_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a call graph for functions and methods.
        
        Args:
            code_elements: List of code elements
            
        Returns:
            Dictionary containing graph data and image
        """
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Filter functions and methods
            functions = [element for element in code_elements 
                       if element['type'] in ('function', 'method')]
            
            # Add nodes for each function
            for function in functions:
                G.add_node(function['name'], type=function['type'])
            
            # Add edges based on function calls
            for function in functions:
                for other_function in functions:
                    if function['name'] != other_function['name'] and other_function['name'] in function['code']:
                        G.add_edge(function['name'], other_function['name'])
            
            # Generate graph image
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes with different colors based on type
            node_colors = {
                'function': 'skyblue',
                'method': 'lightcoral'
            }
            
            for node_type in node_colors:
                nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == node_type]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors[node_type], 
                                     node_size=500, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'graph': G,
                'image': f"data:image/png;base64,{image_base64}",
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error generating call graph: {str(e)}")
            return {
                'error': str(e)
            }
    
    def generate_class_hierarchy(self, code_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a class hierarchy graph.
        
        Args:
            code_elements: List of code elements
            
        Returns:
            Dictionary containing graph data and image
        """
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Filter classes
            classes = [element for element in code_elements if element['type'] == 'class']
            
            # Add nodes for each class
            for class_element in classes:
                G.add_node(class_element['name'])
            
            # Add edges based on inheritance
            for class_element in classes:
                if 'bases' in class_element:
                    for base in class_element['bases']:
                        if base in [c['name'] for c in classes]:
                            G.add_edge(base, class_element['name'])
            
            # Generate graph image
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=500, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'graph': G,
                'image': f"data:image/png;base64,{image_base64}",
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error generating class hierarchy: {str(e)}")
            return {
                'error': str(e)
            }
    
    def generate_module_graph(self, code_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a module dependency graph.
        
        Args:
            code_elements: List of code elements
            
        Returns:
            Dictionary containing graph data and image
        """
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Get unique modules
            modules = {}
            for element in code_elements:
                module_path = os.path.dirname(element['file_path'])
                module_name = os.path.basename(module_path)
                if module_name not in modules:
                    modules[module_name] = {
                        'path': module_path,
                        'elements': []
                    }
                modules[module_name]['elements'].append(element)
            
            # Add nodes for each module
            for module_name in modules:
                G.add_node(module_name, count=len(modules[module_name]['elements']))
            
            # Add edges based on imports
            for module_name, module_data in modules.items():
                for element in module_data['elements']:
                    for other_module_name, other_module_data in modules.items():
                        if module_name != other_module_name:
                            for other_element in other_module_data['elements']:
                                if other_element['name'] in element['code']:
                                    G.add_edge(module_name, other_module_name)
                                    break
            
            # Generate graph image
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes with size based on number of elements
            node_sizes = [G.nodes[n]['count'] * 100 for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_color='lightsalmon', node_size=node_sizes, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Save image to a buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'graph': G,
                'image': f"data:image/png;base64,{image_base64}",
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error generating module graph: {str(e)}")
            return {
                'error': str(e)
            }

