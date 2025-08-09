"""
Local RAG Assistant - A document search and question answering system.

This package provides a complete RAG (Retrieval Augmented Generation) system
with web interface, document management, and AI-powered question answering.
"""

__version__ = "0.1.0"
__author__ = "Joseph Bayley"

from .rag.rag_system import RAGSystem
from .data.document_manager import DocumentManager

__all__ = ["RAGSystem", "DocumentManager"]