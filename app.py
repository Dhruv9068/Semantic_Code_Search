from flask import Flask, render_template, request, jsonify
import os
import json
from code_parser import CodeParser
from embedding_generator import EmbeddingGenerator
from indexer import Indexer
from searcher import Searcher
from config import Config
from utils import setup_logging, format_code_for_display
import logging

# Initialize logging
logger = setup_logging()

app = Flask(__name__)

# Initialize components
config = Config()
code_parser = CodeParser(config)
embedding_generator = EmbeddingGenerator(config)
indexer = Indexer(config)
searcher = Searcher(config)

# Set components for searcher
searcher.set_components(indexer, embedding_generator)

# Check if index exists, if not create empty one
if not os.path.exists(config.index_dir):
    os.makedirs(config.index_dir)
    indexer.initialize_index()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    try:
        query = request.form.get('query', '')
        num_results = int(request.form.get('num_results', 10))
        language_filter = request.form.get('language_filter', '')
        exact_match = request.form.get('exact_match', 'false') == 'true'
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Searching for: {query}")
        results = searcher.search(query, num_results)
        
        # Apply language filter
        if language_filter:
            results = searcher.filter_results(results, language=language_filter)
        
        # Format results for display
        formatted_results = []
        for result in results:
            code_snippet = format_code_for_display(result['code'], result['language'])
            formatted_results.append({
                'file_path': result['file_path'],
                'language': result['language'],
                'code': code_snippet,
                'similarity': f"{result['similarity']:.2f}",
                'line_numbers': result['line_numbers']
            })
        
        return jsonify({'results': formatted_results})
    
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/index', methods=['POST'])
def index_repository():
    """Handle repository indexing requests."""
    try:
        repo_path = request.form.get('repo_path', '')
        index_name = request.form.get('index_name', '')
        index_type = request.form.get('index_type', 'full')
        
        # Advanced options
        chunk_size = request.form.get('chunk_size')
        chunk_overlap = request.form.get('chunk_overlap')
        ignore_patterns = request.form.get('ignore_patterns')
        
        if not repo_path:
            return jsonify({'error': 'Repository path cannot be empty'}), 400
        
        if not os.path.exists(repo_path):
            return jsonify({'error': 'Repository path does not exist'}), 400
        
        logger.info(f"Indexing repository: {repo_path}")
        
        # Apply advanced options if provided
        if chunk_size:
            config.max_chunk_size = int(chunk_size)
        
        if chunk_overlap:
            config.chunk_overlap = int(chunk_overlap)
        
        if ignore_patterns:
            custom_patterns = [pattern.strip() for pattern in ignore_patterns.split('\n') if pattern.strip()]
            if custom_patterns:
                config.ignore_patterns = custom_patterns
        
        # Parse code files
        parsed_files = code_parser.parse_repository(repo_path)
        logger.info(f"Parsed {len(parsed_files)} files")
        
        # Generate embeddings
        embeddings_data = embedding_generator.generate_embeddings(parsed_files)
        logger.info(f"Generated embeddings for {len(embeddings_data)} code snippets")
        
        # Index the embeddings
        if index_type == 'incremental':
            # For incremental updates, we would need to implement a way to update existing embeddings
            # This is a simplified version
            indexer.index_embeddings(embeddings_data)
        else:
            # For full index, we clear the existing index first
            if index_name:
                # In a real implementation, we would support multiple named indexes
                pass
            indexer.index_embeddings(embeddings_data)
        
        logger.info("Indexed embeddings successfully")
        
        return jsonify({'success': True, 'message': f'Indexed {len(parsed_files)} files with {len(embeddings_data)} code snippets'})
    
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the indexed code."""
    try:
        stats = indexer.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-index', methods=['POST'])
def clear_index():
    """Clear the current index."""
    try:
        indexer.clear_index()
        return jsonify({'success': True, 'message': 'Index cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize_repository():
    """Generate visualization data for a repository."""
    try:
        repo = request.form.get('repo', '')
        viz_type = request.form.get('type', 'dependency')
        depth = int(request.form.get('depth', 2))
        
        if not repo:
            return jsonify({'error': 'Repository name cannot be empty'}), 400
        
        # In a real implementation, we would generate actual visualization data
        # This is a placeholder
        visualization_data = {
            'repo': repo,
            'type': viz_type,
            'depth': depth,
            'nodes': [],
            'links': []
        }
        
        return jsonify(visualization_data)
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update application settings."""
    try:
        embedding_model = request.form.get('embedding_model', 'all-MiniLM-L6-v2')
        use_gpu = request.form.get('use_gpu', 'false') == 'true'
        extensions = request.form.getlist('extensions')
        
        # Update config
        config.embedding_model = embedding_model
        config.use_gpu = use_gpu
        
        if extensions:
            config.supported_extensions = extensions
        
        return jsonify({'success': True, 'message': 'Settings updated successfully'})
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=config.debug_mode, host=config.host, port=config.port)

