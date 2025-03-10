import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes natural language queries for code search.
    """
    
    def __init__(self):
        pass
    
    def process_query(self, query: str) -> str:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Processed query
        """
        # Remove extra whitespace
        processed_query = ' '.join(query.split())
        
        # Convert to lowercase
        processed_query = processed_query.lower()
        
        # Add more processing as needed
        
        logger.info(f"Processed query: {processed_query}")
        
        return processed_query
    
    def extract_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract filters from a query.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (processed query, filters dictionary)
        """
        filters = {}
        processed_query = query
        
        # Extract type filter (function, class, method)
        type_keywords = {
            "function": ["function", "def", "functions"],
            "class": ["class", "classes"],
            "method": ["method", "methods"]
        }
        
        for filter_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in query.lower():
                    filters["type"] = filter_type
                    # Remove the keyword from the query
                    processed_query = processed_query.replace(keyword, "")
                    break
            if "type" in filters:
                break
        
        # Process the remaining query
        processed_query = self.process_query(processed_query)
        
        return processed_query, filters

