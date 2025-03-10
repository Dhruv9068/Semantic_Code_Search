from flask import Flask, render_template, request, jsonify
import os
import logging
from typing import Dict, List, Any, Optional
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.repository import Repository
from indexer.embeddings import CodeEmbeddings
from indexer.index_manager import IndexManager
from search.search_engine import SearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
repository = Repository()
embeddings = CodeEmbeddings()
index_manager = IndexManager()
search_engine = SearchEngine()

@app.route('/')
def index():
    """Render the main page."""
    available_indexes = search_engine.get_available_indexes()
    current_index = search_engine.get_current_index_name()
    
    return render_template('index.html', 
                          available_indexes=available_indexes,
                          current_index=current_index)

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    data = request.json
    query = data.get('query', '')
    top_k = int(data.get('top_k', 10))
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    results = search_engine.search(query, top_k)
    
    return jsonify({"results": results})

@app.route('/load_index', methods=['POST'])
def load_index():
    """Load a search index."""
    data = request.json
    index_name = data.get('index_name', '')
    
    if not index_name:
        return jsonify({"error": "Index name is required"}), 400
    
    success = search_engine.load_index(index_name)
    
    if not success:
        return jsonify({"error": f"Failed to load index {index_name}"}), 400
    
    return jsonify({"message": f"Index {index_name} loaded successfully"})

@app.route('/index_repository', methods=['POST'])
def index_repository():
    """Index a repository."""
    data = request.json
    repo_url = data.get('repo_url', '')
    repo_path = data.get('repo_path', '')
    index_name = data.get('index_name', '')
    
    if not index_name:
        return jsonify({"error": "Index name is required"}), 400
    
    if repo_url:
        # Clone and index remote repository
        repo_path = repository.clone_repository(repo_url)
        if not repo_path:
            return jsonify({"error": f"Failed to clone repository {repo_url}"}), 400
    elif repo_path:
        # Index local repository
        repo_path = repository.add_local_repository(repo_path)
        if not repo_path:
            return jsonify({"error": f"Failed to add local repository {repo_path}"}), 400
    else:
        return jsonify({"error": "Either repo_url or repo_path is required"}), 400
    
    # Parse repository
    code_elements = repository.parse_repository(repo_path)
    
    # Generate embeddings
    embeddings_dict = embeddings.generate_embeddings(code_elements)
    
    # Save embeddings
    embeddings.save_embeddings(embeddings_dict, index_name)
    
    # Create index
    index_manager.create_index(embeddings_dict, index_name)
    
    # Load the newly created index
    search_engine.load_index(index_name)
    
    return jsonify({
        "message": f"Repository indexed successfully as {index_name}",
        "stats": {
            "code_elements": len(code_elements)
        }
    })

@app.route('/list_indexes', methods=['GET'])
def list_indexes():
    """List available indexes."""
    available_indexes = search_engine.get_available_indexes()
    
    return jsonify({"indexes": available_indexes})

@app.route('/delete_index', methods=['POST'])
def delete_index():
    """Delete a search index."""
    data = request.json
    index_name = data.get('index_name', '')
    
    if not index_name:
        return jsonify({"error": "Index name is required"}), 400
    
    success = index_manager.delete_index(index_name)
    
    if not success:
        return jsonify({"error": f"Failed to delete index {index_name}"}), 400
    
    return jsonify({"message": f"Index {index_name} deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True)

