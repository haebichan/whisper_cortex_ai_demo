"""Cortex Search service integration for speech-to-text application."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import streamlit as st
from snowflake.core import Root
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException

logger = logging.getLogger(__name__)


class CortexSearchService:
    """Handles Cortex Search operations and LLM processing."""
    
    def __init__(self, cortex_search_service: str = "haebi_cortex_search_service", chunk_limit: int = 2):
        self.cortex_search_service = cortex_search_service
        self.chunk_limit = chunk_limit
        self.session = None
        self.root = None
        self.initialization_error = None
        self._initialize_session()
    
    def _initialize_session(self) -> None:
        """Initialize Snowflake session and root object."""
        try:
            # First, try to get active session (SiS environment)
            self.session = self._try_get_active_session()
            
            if self.session is None:
                # If that fails, try to create session from connection.json (open source)
                self.session = self._try_create_session_from_config()
            
            if self.session is not None:
                self.root = Root(self.session)
                logger.info("Cortex Search service initialized successfully")
                self.initialization_error = None
            else:
                error_msg = "Failed to initialize Snowflake session - no connection method worked"
                logger.error(error_msg)
                self.initialization_error = error_msg
                
        except Exception as e:
            error_msg = f"Failed to initialize Cortex Search service: {e}"
            logger.error(error_msg)
            self.session = None
            self.root = None
            self.initialization_error = str(e)
    
    def _try_get_active_session(self) -> Optional[Session]:
        """Try to get active session (for SiS environment)."""
        try:
            from snowflake.snowpark.context import get_active_session
            session = get_active_session()
            
            # Check if database and schema are properly set
            db = session.get_current_database()
            schema = session.get_current_schema()
            
            if not db or not schema:
                logger.info(f"Active session missing context - DB: {db}, Schema: {schema}. Falling back to connection.json")
                return None
            
            logger.info("Successfully connected using active session (SiS environment)")
            return session
        except Exception as e:
            logger.info(f"Active session not available (likely open source environment): {e}")
            return None
    
    def _try_create_session_from_config(self) -> Optional[Session]:
        """Try to create session from connection.json (for open source environment)."""
        try:
            # Look for connection.json in current directory and parent directories
            config_paths = [
                Path("connection.json"),
                Path("../connection.json"),
                Path("../../connection.json"),
                Path.cwd() / "connection.json",
                Path(__file__).parent / "connection.json",
                Path(__file__).parent.parent / "connection.json"
            ]
            
            connection_parameters = None
            config_file_used = None
            
            for config_path in config_paths:
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            connection_parameters = json.load(f)
                            config_file_used = config_path
                            break
                    except Exception as e:
                        logger.warning(f"Failed to read {config_path}: {e}")
                        continue
            
            if connection_parameters is None:
                logger.warning("No connection.json file found in expected locations")
                return None
            
            # Extract and store cortex search service name if provided
            if 'cortex_search_service' in connection_parameters:
                self.cortex_search_service = connection_parameters.pop('cortex_search_service')
                logger.info(f"Using Cortex Search service from config: {self.cortex_search_service}")
            
            # Create session using connection parameters (without the cortex_search_service key)
            session = Session.builder.configs(connection_parameters).create()
            logger.info(f"Successfully connected using connection.json from {config_file_used}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session from connection.json: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Cortex Search service is available."""
        return self.session is not None and self.root is not None
    
    def get_connection_info(self) -> str:
        """Get information about the current connection."""
        if not self.is_available():
            return f"Not connected. Error: {self.initialization_error or 'Unknown error'}"
        
        try:
            # Try to get basic session info
            current_db = self.session.get_current_database()
            current_schema = self.session.get_current_schema()
            current_warehouse = self.session.get_current_warehouse()
            
            return f"Connected - DB: {current_db}, Schema: {current_schema}, Warehouse: {current_warehouse}"
        except Exception as e:
            return f"Connected (details unavailable: {e})"
    
    def search_and_summarize(self, query: str) -> Tuple[str, bool]:
        """
        Search using Cortex Search and return a summarized response.
        
        Args:
            query: The search query from voice transcription
            
        Returns:
            Tuple of (response_text, success_flag)
        """
        if not self.is_available():
            error_msg = f"Cortex Search service is not available. {self.get_connection_info()}"
            return error_msg, False
        
        try:
            # Query Cortex Search
            search_results, error_msg = self._query_cortex_search(query)
            
            if not search_results:
                if error_msg:
                    logger.error(f"Cortex Search error: {error_msg}")
                    return f"Search error: {error_msg}", False
                else:
                    return "No relevant documents found for your query.", False
            
            # Extract content from search results - handle the exact format from your original code
            content = search_results.get('PAGE_CONTENT') if isinstance(search_results, dict) else getattr(search_results, 'PAGE_CONTENT', None)
            
            if not content:
                # Try alternative ways to extract content
                if hasattr(search_results, '__dict__'):
                    content = str(search_results.__dict__)
                else:
                    content = str(search_results)
            
            if not content:
                return "No content found in search results.", False
            
            # Use Cortex LLM to summarize the results - exactly like your original code
            summary = self._generate_summary(query, content)
            
            if summary:
                return summary, True
            else:
                return "Failed to generate summary from search results.", False
                
        except Exception as e:
            logger.error(f"Error in search_and_summarize: {e}")
            return f"An error occurred during search: {str(e)}", False
    
    def _query_cortex_search(self, query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Query the Cortex Search service.
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (results, error_message)
        """
        try:
            db, schema = self.session.get_current_database(), self.session.get_current_schema()
            
            cortex_search_service = (
                self.root.databases[db]
                .schemas[schema]
                .cortex_search_services[self.cortex_search_service]
            )
            
            # Use the exact same syntax as your original working code
            context_documents = cortex_search_service.search(
                query, columns=[], limit=self.chunk_limit
            )
            
            results = context_documents.results
            
            if results:
                logger.info(f"Found {len(results) if isinstance(results, list) else 1} search results")
                # Return results exactly as they come
                return results, None
            else:
                return None, "No results found"
                
        except Exception as e:
            logger.error(f"Cortex Search query failed: {e}")
            return None, str(e)
    
    def _generate_summary(self, query: str, content: str) -> Optional[str]:
        """
        Generate a summary using Cortex LLM.
        
        Args:
            query: Original user query
            content: Content from search results
            
        Returns:
            Summarized response or None if failed
        """
        try:
            # Escape single quotes in content to prevent SQL injection
            safe_content = content.replace("'", "''")
            safe_query = query.replace("'", "''")
            
            sql_query = f"""
            SELECT snowflake.cortex.complete(
                'llama3.2-1b', 
                concat(
                    'User asked: {safe_query}. ',
                    'The following is the relevant information: ',
                    '{safe_content}',
                    ' Can you provide a clear, concise answer to the user''s question based on this information?'
                )
            ) as RESULT
            """
            
            cortex_input = self.session.sql(sql_query)
            result = cortex_input.collect()
            
            if result and len(result) > 0:
                summary = result[0]["RESULT"]
                logger.info("Successfully generated summary using Cortex LLM")
                return summary
            else:
                logger.warning("No result from Cortex LLM")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the Cortex Search connection.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.is_available():
            error_msg = f"Snowflake session not available. {self.initialization_error or 'Unknown error'}"
            return False, error_msg
        
        try:
            # Test basic session connectivity
            connection_info = self.get_connection_info()
            
            # Just return success if we have a working session and root
            # Don't test the actual search in the connection test
            return True, f"Connection successful. {connection_info}"
                
        except Exception as e:
            return False, f"Connection test error: {str(e)}"