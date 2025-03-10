import os
import re
from typing import Dict, List, Tuple, Any
import logging
from pygments import lexers
from pygments.util import ClassNotFound
import chardet

class CodeParser:
    """
    Parses code files from repositories into manageable chunks for embedding generation.
    """
    
    def __init__(self, config):
        """
        Initialize the CodeParser with configuration.
        
        Args:
            config: Configuration object containing parser settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = config.supported_extensions
        self.max_chunk_size = config.max_chunk_size
        self.min_chunk_size = config.min_chunk_size
        self.overlap_size = config.chunk_overlap
        self.ignore_patterns = config.ignore_patterns
        
    def parse_repository(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Parse all code files in a repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of dictionaries containing parsed code information
        """
        self.logger.info(f"Parsing repository: {repo_path}")
        parsed_files = []
        
        # Check if repo_path is a directory
        if os.path.isdir(repo_path):
            # Process all files in the directory
            for root, _, files in os.walk(repo_path):
                # Skip directories that match ignore patterns
                if any(re.search(pattern, root) for pattern in self.ignore_patterns):
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip files that match ignore patterns
                    if any(re.search(pattern, file) for pattern in self.ignore_patterns):
                        continue
                        
                    # Check if file extension is supported
                    _, ext = os.path.splitext(file)
                    if ext.lower() not in self.supported_extensions:
                        continue
                    
                    try:
                        parsed_file = self.parse_file(file_path, repo_path)
                        if parsed_file:
                            parsed_files.append(parsed_file)
                    except Exception as e:
                        self.logger.error(f"Error parsing file {file_path}: {str(e)}")
        else:
            # Process a single file
            _, ext = os.path.splitext(repo_path)
            if ext.lower() in self.supported_extensions:
                try:
                    # Use the directory containing the file as the repo_path for relative paths
                    dir_path = os.path.dirname(repo_path)
                    parsed_file = self.parse_file(repo_path, dir_path)
                    if parsed_file:
                        parsed_files.append(parsed_file)
                except Exception as e:
                    self.logger.error(f"Error parsing file {repo_path}: {str(e)}")
        
        return parsed_files
    
    def parse_file(self, file_path: str, repo_path: str) -> Dict[str, Any]:
        """
        Parse a single code file.
        
        Args:
            file_path: Path to the file
            repo_path: Base repository path for relative path calculation
            
        Returns:
            Dictionary containing parsed file information
        """
        try:
            # Special handling for Java files
            _, ext = os.path.splitext(file_path)
            is_java_file = ext.lower() == '.java'
            
            # Detect file encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Determine language using pygments
            try:
                lexer = lexers.get_lexer_for_filename(file_path)
                language = lexer.name
            except ClassNotFound:
                # Default to extension-based language detection
                language = ext.lstrip('.').lower()
            
            # Get relative path from repo root
            rel_path = os.path.relpath(file_path, repo_path)
            
            # Create chunks - special handling for Java
            if is_java_file:
                chunks = self.create_java_chunks(content)
            else:
                chunks = self.create_chunks(content, language)
            
            return {
                'file_path': rel_path,
                'absolute_path': file_path,
                'language': language,
                'content': content,
                'chunks': chunks
            }
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def create_java_chunks(self, content: str) -> List[Dict[str, Any]]:
        """
        Special method to create chunks for Java files.
        
        Args:
            content: Java code content
            
        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        lines = content.split('\n')
        
        # First try to identify classes
        class_pattern = r'(public|private|protected)?\s+(class|interface|enum)\s+(\w+)'
        class_matches = list(re.finditer(class_pattern, content))
        
        if class_matches:
            # Process each class
            for i, match in enumerate(class_matches):
                class_start = match.start()
                class_name = match.group(3)
                
                # Find the class end (either the next class or the end of file)
                if i < len(class_matches) - 1:
                    class_end = class_matches[i+1].start()
                else:
                    class_end = len(content)
                
                class_content = content[class_start:class_end]
                
                # Calculate line numbers
                start_line = content[:class_start].count('\n') + 1
                end_line = start_line + class_content.count('\n')
                
                # Add the class as a chunk
                chunks.append({
                    'code': class_content,
                    'line_numbers': (start_line, end_line)
                })
                
                # Try to find methods within the class
                method_pattern = r'(public|private|protected)?\s+(static)?\s+\w+\s+(\w+)\s*$$[^)]*$$\s*\{'
                method_matches = list(re.finditer(method_pattern, class_content))
                
                for j, method_match in enumerate(method_matches):
                    method_start = method_match.start()
                    method_name = method_match.group(3)
                    
                    # Find the method end (either the next method or the end of class)
                    if j < len(method_matches) - 1:
                        method_end = method_matches[j+1].start()
                    else:
                        # Find the closing brace of the method
                        brace_count = 1
                        pos = method_match.end()
                        while brace_count > 0 and pos < len(class_content):
                            if class_content[pos] == '{':
                                brace_count += 1
                            elif class_content[pos] == '}':
                                brace_count -= 1
                            pos += 1
                        method_end = pos
                    
                    method_content = class_content[method_start:method_end]
                    
                    # Calculate line numbers
                    method_start_line = start_line + class_content[:method_start].count('\n')
                    method_end_line = method_start_line + method_content.count('\n')
                    
                    # Add the method as a chunk
                    chunks.append({
                        'code': method_content,
                        'line_numbers': (method_start_line, method_end_line)
                    })
        
        # If no classes were found or chunks are empty, fall back to simple chunking
        if not chunks:
            chunks = self.split_by_size(lines)
        
        return chunks
    
    def create_chunks(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Split code content into semantic chunks.
        
        Args:
            content: Code content as string
            language: Programming language of the code
            
        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        lines = content.split('\n')
        
        # Try to split by functions/classes first
        semantic_chunks = self.split_by_semantic_units(content, lines, language)
        
        if semantic_chunks:
            chunks.extend(semantic_chunks)
        else:
            # Fall back to simple chunking if semantic splitting fails
            chunks.extend(self.split_by_size(lines))
        
        return chunks
    
    def split_by_semantic_units(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """
        Split code by semantic units like functions and classes.
        
        Args:
            content: Full code content
            lines: Code split into lines
            language: Programming language of the code
            
        Returns:
            List of dictionaries containing semantic chunk information
        """
        chunks = []
        
        # Common patterns for function and class definitions based on language
        patterns = []
        
        if language.lower() in ['python', 'py']:
            patterns = [
                r'^\s*(def\s+\w+\s*$$.*?$$:.*?(?=^\s*(?:def|class)\s+\w+|\Z))',  # Python functions
                r'^\s*(class\s+\w+.*?(?=^\s*(?:def|class)\s+\w+|\Z))'  # Python classes
            ]
        elif language.lower() in ['javascript', 'js']:
            patterns = [
                r'^\s*(function\s+\w+\s*$$.*?$$\s*\{.*?^\s*\})',  # JavaScript functions
                r'^\s*(class\s+\w+.*?\{.*?^\s*\})',  # JavaScript classes
                r'^\s*(\w+\s*=\s*function\s*$$.*?$$\s*\{.*?^\s*\})',  # JavaScript function expressions
                r'^\s*(const|let|var)\s+(\w+)\s*=\s*$$.*?$$\s*=>\s*\{.*?^\s*\}'  # JavaScript arrow functions
            ]
        elif language.lower() in ['java']:
            patterns = [
                r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*$$.*?$$\s*\{.*?^\s*\}',  # Java methods
                r'^\s*(public|private|protected)?\s*(class|interface|enum)\s+\w+.*?\{.*?^\s*\}'  # Java classes
            ]
        else:
            # Generic patterns for other languages
            patterns = [
                r'^\s*(function|def|method|procedure)\s+\w+\s*$$.*?$$.*?(?=^\s*(?:function|def|method|procedure|class)\s+\w+|\Z)',
                r'^\s*(class|interface|struct|enum)\s+\w+.*?(?=^\s*(?:function|def|method|procedure|class|interface|struct|enum)\s+\w+|\Z)'
            ]
        
        # Join lines with newlines for regex with DOTALL
        full_content = '\n'.join(lines)
        
        semantic_units = []
        for pattern in patterns:
            matches = re.finditer(pattern, full_content, re.MULTILINE | re.DOTALL)
            for match in matches:
                unit_text = match.group(1)
                start_line = full_content[:match.start()].count('\n') + 1
                end_line = start_line + unit_text.count('\n')
                
                semantic_units.append({
                    'text': unit_text,
                    'start_line': start_line,
                    'end_line': end_line
                })
        
        # If no semantic units found, try a simpler approach
        if not semantic_units:
            # For JavaScript, try to find functions and methods with a simpler approach
            if language.lower() in ['javascript', 'js']:
                # Look for function declarations, function expressions, and methods
                js_patterns = [
                    r'function\s+\w+\s*$$[^)]*$$\s*\{',  # function declarations
                    r'\w+\s*:\s*function\s*$$[^)]*$$\s*\{',  # object methods
                    r'(const|let|var)\s+\w+\s*=\s*function\s*$$[^)]*$$\s*\{',  # function expressions
                    r'(const|let|var)\s+\w+\s*=\s*$$[^)]*$$\s*=>\s*\{',  # arrow functions with block
                    r'(const|let|var)\s+\w+\s*=\s*$$[^)]*$$\s*=>'  # arrow functions without block
                ]
                
                for i, line in enumerate(lines, 1):
                    for pattern in js_patterns:
                        if re.search(pattern, line):
                            # Found a function start, now find its end
                            start_line = i
                            end_line = i
                            brace_count = line.count('{') - line.count('}')
                            
                            # If it's an arrow function without a block, just take a few lines
                            if '=>' in line and '{' not in line:
                                end_line = min(i + 5, len(lines))
                            else:
                                # Otherwise, count braces to find the end
                                for j in range(i, len(lines)):
                                    end_line = j
                                    brace_count += lines[j].count('{') - lines[j].count('}')
                                    if brace_count <= 0:
                                        break
                            
                            unit_text = '\n'.join(lines[start_line-1:end_line])
                            semantic_units.append({
                                'text': unit_text,
                                'start_line': start_line,
                                'end_line': end_line
                            })
            
            # For Java, try to find methods and classes with a simpler approach
            elif language.lower() in ['java']:
                # Look for method and class declarations
                java_patterns = [
                    r'(public|private|protected)?\s+(static)?\s+\w+\s+\w+\s*$$[^)]*$$\s*\{',  # methods
                    r'(public|private|protected)?\s+(class|interface|enum)\s+\w+.*?\{'  # classes
                ]
                
                for i, line in enumerate(lines, 1):
                    for pattern in java_patterns:
                        if re.search(pattern, line):
                            # Found a method/class start, now find its end
                            start_line = i
                            end_line = i
                            brace_count = line.count('{') - line.count('}')
                            
                            for j in range(i, len(lines)):
                                end_line = j
                                brace_count += lines[j].count('{') - lines[j].count('}')
                                if brace_count <= 0:
                                    break
                            
                            unit_text = '\n'.join(lines[start_line-1:end_line])
                            semantic_units.append({
                                'text': unit_text,
                                'start_line': start_line,
                                'end_line': end_line
                            })
        
        # Sort by start line
        semantic_units.sort(key=lambda x: x['start_line'])
        
        # Convert to chunks
        for unit in semantic_units:
            if len(unit['text'].split()) >= self.min_chunk_size:
                chunks.append({
                    'code': unit['text'],
                    'line_numbers': (unit['start_line'], unit['end_line'])
                })
        
        # If no semantic units were found or they're too small, fall back to simple chunking
        if not chunks:
            return self.split_by_size(lines)
        
        return chunks
    
    def split_by_size(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Split code into chunks based on size constraints.
        
        Args:
            lines: Code split into lines
            
        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            line_size = len(line.split()) if line else 0
            
            if current_size + line_size > self.max_chunk_size and current_size >= self.min_chunk_size:
                # Save current chunk
                chunks.append({
                    'code': '\n'.join(current_chunk),
                    'line_numbers': (start_line, i - 1)
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap_size)
                current_chunk = current_chunk[overlap_start:]
                start_line = i - len(current_chunk)
                current_size = sum(len(l.split()) if l else 0 for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the last chunk if it's not empty
        if current_chunk and current_size >= self.min_chunk_size:
            chunks.append({
                'code': '\n'.join(current_chunk),
                'line_numbers': (start_line, len(lines))
            })
        elif current_chunk:  # If it's too small, add it anyway
            chunks.append({
                'code': '\n'.join(current_chunk),
                'line_numbers': (start_line, len(lines))
            })
        
        return chunks

