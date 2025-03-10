import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.parser.repository import Repository
from src.indexer.embeddings import CodeEmbeddings
from src.indexer.index_manager import IndexManager
from src.search.search_engine import SearchEngine
from src.web.app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Semantic Code Search Engine')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a repository')
    index_parser.add_argument('--repo-url', help='URL of the Git repository')
    index_parser.add_argument('--repo-path', help='Path to the local repository')
    index_parser.add_argument('--name', required=True, help='Name of the index')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for code')
    search_parser.add_argument('--index', required=True, help='Name of the index to search')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    
    # List command
    subparsers.add_parser('list', help='List available indexes')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start the web interface')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == 'index':
        # Index a repository
        repository = Repository()
        embeddings = CodeEmbeddings()
        index_manager = IndexManager()
        
        if args.repo_url:
            # Clone and index remote repository
            repo_path = repository.clone_repository(args.repo_url)
            if not repo_path:
                logger.error(f"Failed to clone repository {args.repo_url}")
                return 1
        elif args.repo_path:
            # Index local repository
            repo_path = repository.add_local_repository(args.repo_path)
            if not repo_path:
                logger.error(f"Failed to add local repository {args.repo_path}")
                return 1
        else:
            logger.error("Either --repo-url or --repo-path is required")
            return 1
        
        # Parse repository
        code_elements = repository.parse_repository(repo_path)
        
        # Generate embeddings
        embeddings_dict = embeddings.generate_embeddings(code_elements)
        
        # Save embeddings
        embeddings.save_embeddings(embeddings_dict, args.name)
        
        # Create index
        index_manager.create_index(embeddings_dict, args.name)
        
        logger.info(f"Repository indexed successfully as {args.name}")
        logger.info(f"Found {len(code_elements)} code elements")
        
    elif args.command == 'search':
        # Search for code
        search_engine = SearchEngine()
        
        # Load index
        if not search_engine.load_index(args.index):
            logger.error(f"Failed to load index {args.index}")
            return 1
        
        # Search
        results = search_engine.search(args.query, args.top_k)
        
        # Display results
        print(f"Found {len(results)} results for query: {args.query}")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Type: {result['type']}")
            print(f"Name: {result['name']}")
            print(f"Score: {result['score']:.2f}")
            print(f"File: {result['file_path']}")
            print(f"Line: {result['line_number']}")
            
            if result['docstring']:
                print(f"\nDocstring:\n{result['docstring']}")
            
            print(f"\nCode:\n{result['code']}")
        
    elif args.command == 'list':
        # List available indexes
        index_manager = IndexManager()
        indexes = index_manager.list_indexes()
        
        if not indexes:
            print("No indexes available")
        else:
            print("Available indexes:")
            for index in indexes:
                print(f"- {index}")
        
    elif args.command == 'web':
        # Start the web interface
        app.run(host=args.host, port=args.port)
        
    else:
        # No command specified
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

