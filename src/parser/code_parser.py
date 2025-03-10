import ast
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeParser:
    """
    Parser for Python code files that extracts functions, classes, and their docstrings.
    """
    
    def __init__(self):
        self.stats = {
            "files_processed": 0,
            "functions_extracted": 0,
            "classes_extracted": 0,
            "methods_extracted": 0
        }
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a Python file and extract code elements.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of dictionaries containing code elements
        """
        if not file_path.endswith('.py'):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            self.stats["files_processed"] += 1
            
            code_elements = []
            
            # Extract module-level docstring
            if ast.get_docstring(tree):
                code_elements.append({
                    'type': 'module',
                    'name': os.path.basename(file_path),
                    'docstring': ast.get_docstring(tree),
                    'code': content,
                    'file_path': file_path,
                    'line_number': 1
                })
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._extract_function(node, file_path, content, code_elements)
                elif isinstance(node, ast.ClassDef):
                    self._extract_class(node, file_path, content, code_elements)
            
            return code_elements
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str, 
                         content: str, code_elements: List[Dict[str, Any]]) -> None:
        """Extract function details from AST node."""
        docstring = ast.get_docstring(node) or ""
        
        # Get the source code for this function
        start_line = node.lineno
        end_line = max(node.body[-1].lineno if node.body else start_line, 
                       start_line + len(docstring.split('\n')) if docstring else start_line)
        
        function_code = "\n".join(content.split("\n")[start_line-1:end_line])
        
        code_elements.append({
            'type': 'function',
            'name': node.name,
            'docstring': docstring,
            'code': function_code,
            'file_path': file_path,
            'line_number': start_line,
            'args': [arg.arg for arg in node.args.args],
            'returns': self._extract_return_annotation(node)
        })
        
        self.stats["functions_extracted"] += 1
    
    def _extract_class(self, node: ast.ClassDef, file_path: str, 
                      content: str, code_elements: List[Dict[str, Any]]) -> None:
        """Extract class details from AST node."""
        docstring = ast.get_docstring(node) or ""
        
        # Get the source code for this class
        start_line = node.lineno
        end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')], default=start_line)
        
        class_code = "\n".join(content.split("\n")[start_line-1:end_line])
        
        class_element = {
            'type': 'class',
            'name': node.name,
            'docstring': docstring,
            'code': class_code,
            'file_path': file_path,
            'line_number': start_line,
            'bases': [self._get_name(base) for base in node.bases]
        }
        
        code_elements.append(class_element)
        self.stats["classes_extracted"] += 1
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_docstring = ast.get_docstring(item) or ""
                
                # Get the source code for this method
                method_start_line = item.lineno
                method_end_line = max(item.body[-1].lineno if item.body else method_start_line,
                                     method_start_line + len(method_docstring.split('\n')) if method_docstring else method_start_line)
                
                method_code = "\n".join(content.split("\n")[method_start_line-1:method_end_line])
                
                code_elements.append({
                    'type': 'method',
                    'name': f"{node.name}.{item.name}",
                    'docstring': method_docstring,
                    'code': method_code,
                    'file_path': file_path,
                    'line_number': method_start_line,
                    'class_name': node.name,
                    'args': [arg.arg for arg in item.args.args],
                    'returns': self._extract_return_annotation(item)
                })
                
                self.stats["methods_extracted"] += 1
    
    def _extract_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if available."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return self._get_name(node.returns)
            elif isinstance(node.returns, ast.Subscript):
                return self._get_name(node.returns)
        return None
    
    def _get_name(self, node: ast.AST) -> str:
        """Get the string representation of a name node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the parsing process."""
        return self.stats

