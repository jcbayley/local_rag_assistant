#!/usr/bin/env python3
"""
RAG Database CLI Utility

Command line utility for adding documents to ChromaDB collections.
Supports adding single files or entire directories.

Usage:
    rag-database -db database_name -c collection_name -d directory
    rag-database -db database_name -c collection_name -f filename
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

from local_rag_assistant.data.document_manager import DocumentManager


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.pdf', '.txt', '.md', '.docx', '.doc', '.csv', '.html', '.htm', 
            '.pptx', '.ppt', '.xlsx', '.xls', '.py', '.cpp', '.cc', '.cxx', 
            '.c', '.h', '.hpp', '.hxx']


def is_supported_file(file_path: Path) -> bool:
    """Check if file extension is supported."""
    return file_path.suffix.lower() in get_supported_extensions()


def scan_directory(directory: Union[str, Path]) -> List[Path]:
    """
    Recursively scan directory for supported document files.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        List of supported file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    supported_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and is_supported_file(file_path):
            supported_files.append(file_path)
    
    return sorted(supported_files)


def process_files(doc_manager: DocumentManager, files: List[Path]) -> None:
    """
    Process list of files and add them to ChromaDB.
    
    Args:
        doc_manager: DocumentManager instance
        files: List of file paths to process
    """
    if not files:
        print("No supported files found to process.")
        return
    
    print(f"Found {len(files)} supported files to process:")
    for file_path in files:
        print(f"  - {file_path}")
    
    print("\nProcessing files...")
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        
        try:
            success, message = doc_manager.add_to_chromadb(file_path)
            if success:
                print(f"  ✓ {message}")
                successful += 1
            else:
                print(f"  ✗ {message}")
                failed += 1
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Total: {len(files)} files")


def get_database_path(database_name: str) -> str:
    """
    Get the full path for a database name.
    Creates a consistent location that the app can locate.
    
    Args:
        database_name: Name of the database
        
    Returns:
        Full path to the database directory
    """
    # Use current working directory as base, but create a standard structure
    base_dir = Path.cwd()
    
    # If database_name doesn't end with _db, add it for consistency
    if not database_name.endswith('_db'):
        database_name = f"{database_name}_db"
    
    db_path = base_dir / database_name
    
    # Create the directory if it doesn't exist
    db_path.mkdir(exist_ok=True)
    
    return str(db_path)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="RAG Database CLI - Add documents to ChromaDB collections",
        epilog=f"Supported file types: {', '.join(get_supported_extensions())}"
    )
    
    parser.add_argument('-db', '--database', required=True,
                       help='Database name (will be created in current directory as <name>_db)')
    
    parser.add_argument('-c', '--collection', required=True,
                       help='Collection name within the database')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--directory',
                      help='Directory to scan for supported documents (recursive)')
    
    group.add_argument('-f', '--file',
                      help='Single file to add to the database')
    
    parser.add_argument('--list-supported', action='store_true',
                       help='List supported file types and exit')
    
    parser.add_argument('--status', action='store_true',
                       help='Show database status and available collections')
    
    args = parser.parse_args()
    
    if args.list_supported:
        print("Supported file types:")
        for ext in get_supported_extensions():
            print(f"  {ext}")
        return
    
    # Get database path
    db_path = get_database_path(args.database)
    print(f"Using database: {db_path}")
    print(f"Collection: {args.collection}")
    
    # Initialize DocumentManager
    try:
        doc_manager = DocumentManager(
            chromadb_path=db_path,
            collection_name=args.collection
        )
        
        if args.status:
            status = doc_manager.get_status()
            print(f"\n=== Database Status ===")
            print(f"ChromaDB Available: {'✓' if status['chromadb_available'] else '✗'}")
            print(f"Database Path: {status['chromadb_path']}")
            print(f"Current Collection: {status['collection_name']}")
            print(f"Embedding Model: {status['embedding_model']}")
            print(f"Available Collections: {', '.join(status['available_collections'])}")
            return
            
    except Exception as e:
        print(f"Error initializing DocumentManager: {e}")
        sys.exit(1)
    
    # Process files
    try:
        if args.directory:
            print(f"Scanning directory: {args.directory}")
            files = scan_directory(args.directory)
            process_files(doc_manager, files)
            
        elif args.file:
            file_path = Path(args.file)
            
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
            
            if not is_supported_file(file_path):
                print(f"Error: Unsupported file type: {file_path.suffix}")
                print(f"Supported types: {', '.join(get_supported_extensions())}")
                sys.exit(1)
            
            process_files(doc_manager, [file_path])
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()