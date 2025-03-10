import os
import git
import logging
import shutil
from typing import List, Dict, Any, Optional
from .code_parser import CodeParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Repository:
    """
    Handles repository operations like cloning, updating, and parsing.
    """
    
    def __init__(self, repo_dir: str = "data/repositories"):
        self.repo_dir = repo_dir
        self.parser = CodeParser()
        
        # Create repository directory if it doesn't exist
        os.makedirs(repo_dir, exist_ok=True)
    
    def clone_repository(self, repo_url: str, branch: str = "main") -> Optional[str]:
        """
        Clone a Git repository.
        
        Args:
            repo_url: URL of the Git repository
            branch: Branch to clone (default: main)
            
        Returns:
            Path to the cloned repository or None if failed
        """
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(self.repo_dir, repo_name)
        
        # Remove existing repository if it exists
        if os.path.exists(repo_path):
            logger.info(f"Repository {repo_name} already exists. Removing...")
            shutil.rmtree(repo_path)
        
        try:
            logger.info(f"Cloning repository {repo_url}...")
            git.Repo.clone_from(repo_url, repo_path, branch=branch)
            logger.info(f"Repository cloned to {repo_path}")
            return repo_path
        except Exception as e:
            logger.error(f"Error cloning repository {repo_url}: {str(e)}")
            return None
    
    def add_local_repository(self, local_path: str) -> Optional[str]:
        """
        Add a local directory as a repository.
        
        Args:
            local_path: Path to the local directory
            
        Returns:
            Path to the repository in the repo_dir or None if failed
        """
        if not os.path.exists(local_path):
            logger.error(f"Local path {local_path} does not exist")
            return None
        
        repo_name = os.path.basename(local_path)
        repo_path = os.path.join(self.repo_dir, repo_name)
        
        # Remove existing repository if it exists
        if os.path.exists(repo_path):
            logger.info(f"Repository {repo_name} already exists. Removing...")
            shutil.rmtree(repo_path)
        
        try:
            logger.info(f"Copying local directory {local_path} to {repo_path}...")
            shutil.copytree(local_path, repo_path)
            logger.info(f"Local directory copied to {repo_path}")
            return repo_path
        except Exception as e:
            logger.error(f"Error copying local directory {local_path}: {str(e)}")
            return None
    
    def parse_repository(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Parse all Python files in a repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of dictionaries containing code elements
        """
        if not os.path.exists(repo_path):
            logger.error(f"Repository path {repo_path} does not exist")
            return []
        
        logger.info(f"Parsing repository {repo_path}...")
        code_elements = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_elements = self.parser.parse_file(file_path)
                    code_elements.extend(file_elements)
        
        logger.info(f"Parsed {len(code_elements)} code elements from repository {repo_path}")
        logger.info(f"Stats: {self.parser.get_stats()}")
        
        return code_elements

