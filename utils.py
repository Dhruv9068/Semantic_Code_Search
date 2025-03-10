import os
import logging
import re
from typing import Tuple
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    log_file = os.path.join(log_dir, "semantic_search.log")
    
    logger = logging.getLogger("semantic_code_search")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def format_code_for_display(code: str, language: str = None) -> str:
    """
    Format code for display with syntax highlighting.
    
    Args:
        code: Code snippet
        language: Programming language of the code
        
    Returns:
        HTML-formatted code with syntax highlighting
    """
    try:
        # Try to get lexer by language name
        if language:
            lexer = get_lexer_by_name(language.lower(), stripall=True)
        else:
            # Guess lexer based on code content
            lexer = guess_lexer(code)
        
        # Format code with HTML formatter
        formatter = HtmlFormatter(linenos=True, cssclass="source")
        html_code = highlight(code, lexer, formatter)
        
        return html_code
    except ClassNotFound:
        # If lexer not found, return code with basic HTML formatting
        return f"<pre><code>{code}</code></pre>"

def extract_repo_name(repo_path: str) -> str:
    """
    Extract repository name from path.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        Repository name
    """
    # Get the last directory name in the path
    return os.path.basename(os.path.normpath(repo_path))

def parse_line_range(line_range: str) -> Tuple[int, int]:
    """
    Parse line range string into start and end line numbers.
    
    Args:
        line_range: String in format "start-end"
        
    Returns:
        Tuple of (start_line, end_line)
    """
    try:
        if '-' in line_range:
            start, end = line_range.split('-')
            return int(start), int(end)
        else:
            line = int(line_range)
            return line, line
    except ValueError:
        return 1, 1  # Default to first line if parsing fails

