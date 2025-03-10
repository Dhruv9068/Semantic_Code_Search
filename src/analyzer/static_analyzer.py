import ast
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StaticAnalyzer:
    """
    Performs static analysis on Python code to detect issues and potential bugs.
    """
    
    def __init__(self):
        pass
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze Python code for potential issues.
        
        Args:
            code: Python code to analyze
            file_path: Optional path to the file (for reference)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            tree = ast.parse(code)
            
            # Collect various issues
            issues = []
            issues.extend(self._check_unused_imports(tree))
            issues.extend(self._check_unused_variables(tree))
            issues.extend(self._check_undefined_variables(tree))
            issues.extend(self._check_complex_functions(tree))
            issues.extend(self._check_too_many_arguments(tree))
            issues.extend(self._check_too_many_locals(tree))
            issues.extend(self._check_too_many_branches(tree))
            issues.extend(self._check_too_many_statements(tree))
            issues.extend(self._check_bare_except(tree))
            issues.extend(self._check_broad_except(tree))
            
            # Group issues by type
            issues_by_type = {}
            for issue in issues:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            return {
                'issues': issues,
                'issues_by_type': issues_by_type,
                'issue_count': len(issues),
                'has_issues': len(issues) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                'error': str(e),
                'issues': [],
                'issues_by_type': {},
                'issue_count': 0,
                'has_issues': False
            }
    
    def _check_unused_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for unused imports."""
        issues = []
        
        # Collect all imported names
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imported_names.add(name.name if name.asname is None else name.asname)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imported_names.add(name.name if name.asname is None else name.asname)
        
        # Collect all used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Find unused imports
        unused_imports = imported_names - used_names
        
        for name in unused_imports:
            issues.append({
                'type': 'unused_import',
                'message': f"Unused import: {name}",
                'name': name
            })
        
        return issues
    
    def _check_unused_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for unused variables."""
        issues = []
        
        # Collect all defined variables
        defined_vars = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                        defined_vars[target.id] = target.lineno
            elif isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    defined_vars[arg.arg] = node.lineno
        
        # Collect all used variables
        used_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
        
        # Find unused variables (excluding special names)
        special_names = {'self', '_', '__', 'cls'}
        unused_vars = {name: lineno for name, lineno in defined_vars.items() 
                      if name not in used_vars and not name.startswith('_') and name not in special_names}
        
        for name, lineno in unused_vars.items():
            issues.append({
                'type': 'unused_variable',
                'message': f"Unused variable: {name}",
                'name': name,
                'lineno': lineno
            })
        
        return issues
    
    def _check_undefined_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for potentially undefined variables."""
        issues = []
        
        # This is a simplified implementation that doesn't account for scopes
        # In a real implementation, we would need to track variables by scope
        
        # Collect all defined variables
        defined_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                        defined_vars.add(target.id)
            elif isinstance(node, ast.FunctionDef):
                defined_vars.add(node.name)
                for arg in node.args.args:
                    defined_vars.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined_vars.add(node.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    defined_vars.add(name.name if name.asname is None else name.asname)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    defined_vars.add(name.name if name.asname is None else name.asname)
        
        # Add built-in names
        defined_vars.update([
            'True', 'False', 'None', 'print', 'len', 'range', 'enumerate', 'zip',
            'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool',
            'sum', 'min', 'max', 'sorted', 'reversed', 'open', 'isinstance',
            'Exception', 'TypeError', 'ValueError', 'KeyError', 'IndexError'
        ])
        
        # Collect all used variables
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined_vars:
                    issues.append({
                        'type': 'undefined_variable',
                        'message': f"Potentially undefined variable: {node.id}",
                        'name': node.id,
                        'lineno': node.lineno
                    })
        
        return issues
    
    def _check_complex_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for overly complex functions."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count the number of statements
                statement_count = sum(1 for _ in ast.walk(node) if isinstance(_, (
                    ast.Assign, ast.AugAssign, ast.Return, ast.Raise, ast.Assert,
                    ast.Import, ast.ImportFrom, ast.If, ast.For, ast.While, ast.Try
                )))
                
                # Count the number of branches
                branch_count = sum(1 for _ in ast.walk(node) if isinstance(_, (
                    ast.If, ast.For, ast.While, ast.Try
                )))
                
                # Check if the function is too complex
                if statement_count > 50:
                    issues.append({
                        'type': 'complex_function',
                        'message': f"Function '{node.name}' has too many statements ({statement_count})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'statement_count': statement_count
                    })
                
                if branch_count > 10:
                    issues.append({
                        'type': 'complex_function',
                        'message': f"Function '{node.name}' has too many branches ({branch_count})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'branch_count': branch_count
                    })
        
        return issues
    
    def _check_too_many_arguments(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for functions with too many arguments."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_count = len(node.args.args)
                
                # Exclude 'self' for methods
                if arg_count > 0 and node.args.args[0].arg == 'self':
                    arg_count -= 1
                
                if arg_count > 5:
                    issues.append({
                        'type': 'too_many_arguments',
                        'message': f"Function '{node.name}' has too many arguments ({arg_count})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'arg_count': arg_count
                    })
        
        return issues
    
    def _check_too_many_locals(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for functions with too many local variables."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count local variables
                local_vars = set()
                
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Assign):
                        for target in subnode.targets:
                            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                                local_vars.add(target.id)
                
                if len(local_vars) > 15:
                    issues.append({
                        'type': 'too_many_locals',
                        'message': f"Function '{node.name}' has too many local variables ({len(local_vars)})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'local_count': len(local_vars)
                    })
        
        return issues
    
    def _check_too_many_branches(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for functions with too many branches."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count branches
                branch_count = 0
                
                for subnode in ast.walk(node):
                    if isinstance(subnode, (ast.If, ast.For, ast.While)):
                        branch_count += 1
                
                if branch_count > 10:
                    issues.append({
                        'type': 'too_many_branches',
                        'message': f"Function '{node.name}' has too many branches ({branch_count})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'branch_count': branch_count
                    })
        
        return issues
    
    def _check_too_many_statements(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for functions with too many statements."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count statements
                statement_count = 0
                
                for subnode in ast.walk(node):
                    if isinstance(subnode, (ast.Assign, ast.AugAssign, ast.Return, ast.Raise,
                                          ast.Assert, ast.If, ast.For, ast.While, ast.Try)):
                        statement_count += 1
                
                if statement_count > 50:
                    issues.append({
                        'type': 'too_many_statements',
                        'message': f"Function '{node.name}' has too many statements ({statement_count})",
                        'name': node.name,
                        'lineno': node.lineno,
                        'statement_count': statement_count
                    })
        
        return issues
    
    def _check_bare_except(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for bare except clauses."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append({
                            'type': 'bare_except',
                            'message': "Bare except clause",
                            'lineno': handler.lineno
                        })
        
        return issues
    
    def _check_broad_except(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for overly broad except clauses."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is not None and isinstance(handler.type, ast.Name) and handler.type.id == 'Exception':
                        issues.append({
                            'type': 'broad_except',
                            'message': "Overly broad except clause (Exception)",
                            'lineno': handler.lineno
                        })
        
        return issues

