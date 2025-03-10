import os
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory {directory}")
        os.makedirs(directory, exist_ok=True)

def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension
    """
    return os.path.splitext(file_path)[1]

def is_python_file(file_path: str) -> bool:
    """
    Check if a file is a Python file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a Python file, False otherwise
    """
    return get_file_extension(file_path) == '.py'

def format_code_for_display(code: str) -> str:
    """
    Format code for display in the web interface.
    
    Args:
        code: Code to format
        
    Returns:
        Formatted code
    """
    # Replace tabs with spaces
    code = code.replace('\t', '    ')
    
    # Ensure the code ends with a newline
    if not code.endswith('\n'):
        code += '\n'
    
    return code

