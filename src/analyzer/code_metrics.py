import ast
import os
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeMetrics:
    """
    Analyzes Python code to extract various complexity and quality metrics.
    """
    
    def __init__(self):
        self.metrics_cache = {}  # Cache for computed metrics
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze Python code and compute various metrics.
        
        Args:
            code: Python code to analyze
            file_path: Optional path to the file (for reference)
            
        Returns:
            Dictionary containing various code metrics
        """
        # Check cache first
        cache_key = hash(code)
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        try:
            tree = ast.parse(code)
            
            metrics = {
                'loc': self._count_lines_of_code(code),
                'cyclomatic_complexity': self._compute_cyclomatic_complexity(tree),
                'halstead_metrics': self._compute_halstead_metrics(code),
                'maintainability_index': 0,  # Will be computed later
                'comment_ratio': self._compute_comment_ratio(code),
                'function_count': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef)),
                'class_count': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef)),
                'average_function_length': self._compute_average_function_length(tree, code),
                'max_nesting_depth': self._compute_max_nesting_depth(tree),
                'cognitive_complexity': self._compute_cognitive_complexity(tree),
                'variable_count': self._count_variables(tree),
                'import_count': self._count_imports(tree),
                'todo_count': code.lower().count('# todo'),
                'fixme_count': code.lower().count('# fixme'),
            }
            
            # Compute maintainability index
            metrics['maintainability_index'] = self._compute_maintainability_index(
                metrics['loc'], 
                metrics['cyclomatic_complexity'],
                metrics['halstead_metrics']['volume']
            )
            
            # Cache the result
            self.metrics_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                'error': str(e),
                'loc': self._count_lines_of_code(code),
                'comment_ratio': self._compute_comment_ratio(code),
            }
    
    def _count_lines_of_code(self, code: str) -> int:
        """Count non-empty, non-comment lines of code."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        return len(non_empty_lines)
    
    def _compute_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """
        Compute the cyclomatic complexity of the code.
        
        McCabe's cyclomatic complexity is defined as:
        M = E - N + 2P
        where:
        - E = number of edges in the control flow graph
        - N = number of nodes in the control flow graph
        - P = number of connected components
        
        For a single function, this simplifies to:
        M = 1 + number of decision points
        """
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # Start with 1 for the function itself
            
            def visit_If(self, node):
                self.complexity += 1
                # Count 'elif' branches
                for handler in node.orelse:
                    if isinstance(handler, ast.If):
                        self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Count each except handler
                self.complexity += len(node.handlers)
                self.generic_visit(node)
            
            def visit_BoolOp(self, node):
                # Count boolean operations (and, or)
                if isinstance(node.op, (ast.And, ast.Or)):
                    self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _compute_halstead_metrics(self, code: str) -> Dict[str, float]:
        """
        Compute Halstead complexity metrics.
        
        Halstead metrics are based on the number of operators and operands in the code.
        """
        # This is a simplified implementation
        operators = set(['=', '+', '-', '*', '/', '%', '**', '//', '==', '!=', '<', '>', '<=', '>=', 
                        'and', 'or', 'not', 'in', 'is', '+=', '-=', '*=', '/=', '%=', '**=', '//='])
        
        # Count operators and operands
        n1 = 0  # Number of unique operators
        n2 = 0  # Number of unique operands
        N1 = 0  # Total number of operators
        N2 = 0  # Total number of operands
        
        # Very simplified counting - in a real implementation, we would use the AST
        words = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']', ' ').split()
        
        operator_count = defaultdict(int)
        operand_count = defaultdict(int)
        
        for word in words:
            if word in operators:
                operator_count[word] += 1
                N1 += 1
            else:
                # This is a very simplified approach - in reality, we'd need to parse the code properly
                operand_count[word] += 1
                N2 += 1
        
        n1 = len(operator_count)
        n2 = len(operand_count)
        
        # Avoid division by zero
        if n1 == 0 or n2 == 0:
            return {
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'time': 0,
                'bugs': 0
            }
        
        # Calculate Halstead metrics
        N = N1 + N2
        n = n1 + n2
        
        volume = N * math.log2(n) if n > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        time = effort / 18  # Time to program (in seconds)
        bugs = volume / 3000  # Estimated number of bugs
        
        return {
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort,
            'time': time,
            'bugs': bugs
        }
    
    def _compute_maintainability_index(self, loc: int, complexity: int, volume: float) -> float:
        """
        Compute the maintainability index.
        
        The maintainability index is a software metric that indicates how maintainable (easy to support and change) the source code is.
        """
        # Original formula: 171 - 5.2 * ln(volume) - 0.23 * complexity - 16.2 * ln(loc)
        # Normalized to 0-100 scale
        
        if loc == 0 or volume == 0:
            return 100  # Perfect score for empty code
        
        raw_mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)
        normalized_mi = max(0, (raw_mi * 100) / 171)
        return normalized_mi
    
    def _compute_comment_ratio(self, code: str) -> float:
        """Compute the ratio of comments to code."""
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len(lines)
        return comment_lines / total_lines if total_lines > 0 else 0
    
    def _compute_average_function_length(self, tree: ast.AST, code: str) -> float:
        """Compute the average length of functions in the code."""
        function_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get the source code for this function
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    function_length = node.end_lineno - node.lineno + 1
                    function_lengths.append(function_length)
        
        return sum(function_lengths) / len(function_lengths) if function_lengths else 0
    
    def _compute_max_nesting_depth(self, tree: ast.AST) -> int:
        """Compute the maximum nesting depth in the code."""
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
            
            def visit_FunctionDef(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_ClassDef(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_If(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_For(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_While(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_Try(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
        
        visitor = NestingVisitor()
        visitor.visit(tree)
        return visitor.max_depth
    
    def _compute_cognitive_complexity(self, tree: ast.AST) -> int:
        """
        Compute the cognitive complexity of the code.
        
        Cognitive complexity is a measure of how difficult it is to understand the control flow of the code.
        """
        class CognitiveComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
            
            def visit_FunctionDef(self, node):
                old_nesting_level = self.nesting_level
                self.nesting_level = 0  # Reset nesting level for each function
                self.generic_visit(node)
                self.nesting_level = old_nesting_level
            
            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level  # Base cost + nesting level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
            
            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
            
            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
            
            def visit_Try(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
            
            def visit_BoolOp(self, node):
                # Add complexity for boolean operations
                if isinstance(node.op, (ast.And, ast.Or)):
                    self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = CognitiveComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _count_variables(self, tree: ast.AST) -> int:
        """Count the number of variables defined in the code."""
        variables = set()
        
        for node in ast.walk(tree):
            # Variable assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
            
            # Function parameters
            elif isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    variables.add(arg.arg)
            
            # For loop variables
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    variables.add(node.target.id)
        
        return len(variables)
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count the number of imports in the code."""
        import_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_count += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                import_count += len(node.names)
        
        return import_count

