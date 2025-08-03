#!/usr/bin/env python3
"""
Web Scraping Tool for RAG System

Command-line tool based on existing scrapy_scrape.py that scrapes websites 
and adds content directly to ChromaDB using the DocumentManager class.

Usage:
    python scrape_web.py --url https://example.com --dbname my_database
    python scrape_web.py --url https://docs.python.org --max-pages 50 --verbose
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
import argparse
import os
import sys
import json
import uuid
import tempfile
from lxml import html
import re
from readability import Document

from document_manager import DocumentManager


class RAGTextSpider(CrawlSpider):
    """
    Scrapy spider based on existing TextSpider2 but integrated with DocumentManager.
    """
    name = 'rag_text_spider'
    
    def __init__(self, start_url=None, doc_manager=None, max_pages=10, verbose=False, *args, **kwargs):
        super(RAGTextSpider, self).__init__(*args, **kwargs)
        
        if not start_url:
            raise ValueError("start_url is required")
        
        self.start_urls = [start_url]
        self.doc_manager = doc_manager
        self.max_pages = int(max_pages)
        self.verbose = verbose
        self.pages_scraped = 0
        self.total_chunks = 0
        
        # Set allowed domains based on start URL
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        
        # Configure rules similar to TextSpider2
        self.rules = (
            Rule(
                LinkExtractor(
                    allow_domains=self.allowed_domains,
                    deny_extensions=[
                        'pdf', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar', 'exe',
                        'csv', 'ppt', 'pptx', 'jpg', 'jpeg', 'png', 'gif', 'svg', 
                        'mp4', 'mp3', 'avi', 'mov', 'wmv', 'flv', 'css', 'js'
                    ],
                    deny=r'(^mailto:|^tel:|\.vcf$|vcard)',
                ),
                callback='parse_item',
                follow=True
            ),
        )
        
        self.visited_urls = set()
        
        # Common strings to remove (from existing scrapy_scrape.py)
        self.replace_strings = [
            "Skip to main content", 
            "The University of Glasgow uses cookies for analytics.",
            "Find out more about our Privacy policy",
            "privacy settings accept",
            "We use cookies",
            "Necessary cookies",
            "Necessary cookies enable core functionality.",
            "The website cannot function properly without these cookies, and can only be disabled by changing your browser preferences.",
            "Analytical cookies help us improve our website.",
            "We use Google Analytics.",
            "All data is anonymised.",
            "Switch analytics ON OFF",
            "Clarity helps us to understand our users' behaviour by visually representing their clicks, taps and scrolling.",
            "Switch clarity ON OFF",
            "Privacy policy close",
            "Study Research Explore Connect",
            "Search icon Close menu icon Menu icon bar 1 Menu icon bar 2 Menu icon bar 3",
            "Cookie banner", "Cookie policy", "Accept cookies", "Manage cookies"
        ]
        
        # Re-compile rules after setting them
        self._compile_rules()

    def parse_item(self, response):
        """Parse individual page - based on TextSpider2 logic."""
        
        # Check if we've hit our page limit
        if self.pages_scraped >= self.max_pages:
            return
        
        url = response.url
        
        # Skip already visited pages
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        
        # Check if content-type is HTML
        content_type = response.headers.get('Content-Type', b'').decode().lower()
        if 'text/html' not in content_type:
            return  # skip non-HTML responses
        
        try:
            # Use Readability to get main content (like TextSpider)
            doc = Document(response.text)
            simplified_html = doc.summary()
            parsed = html.fromstring(simplified_html)
            page_text = parsed.text_content()
            
            # Fallback to manual parsing if readability fails
            if not page_text or len(page_text) < 50:
                # Parse HTML using lxml (like TextSpider2)
                tree = html.fromstring(response.text)
                
                # Remove unwanted elements
                for xpath in [
                    '//script', '//style', '//nav', '//footer', '//header', '//aside',
                    '//*[contains(@class, "cookie")]', '//*[contains(@id, "cookie")]',
                    '//*[contains(@class, "banner")]', '//*[contains(@class, "popup")]'
                ]:
                    for el in tree.xpath(xpath):
                        el.drop_tree()
                
                # Extract visible body text
                text_elements = tree.xpath('//body//text()')
                page_text = '\n'.join(text_elements)
            
            # Clean whitespace
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            
            # Remove common unwanted strings
            for replace_string in self.replace_strings:
                page_text = page_text.replace(replace_string, "").strip()
            
            # Skip if content is too short
            if len(page_text) < 100:
                if self.verbose:
                    print(f"Skipping {url}: content too short ({len(page_text)} chars)")
                return
            
            if self.verbose:
                print(f"Scraping URL: {url} | Text Length: {len(page_text)}")
            
            # Add to ChromaDB using DocumentManager
            if self.doc_manager and self.doc_manager.chromadb_collection:
                try:
                    # Split into chunks
                    chunks = self.doc_manager._chunk_text(page_text, max_tokens=500)
                    
                    # Add each chunk to ChromaDB
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"web-{uuid.uuid4()}"
                        
                        self.doc_manager.chromadb_collection.add(
                            documents=[chunk],
                            metadatas=[{
                                'url': url,
                                'source_type': 'web',
                                'chunk_index': i,
                                'domain': urlparse(url).netloc,
                                'total_chunks': len(chunks)
                            }],
                            ids=[chunk_id]
                        )
                    
                    self.pages_scraped += 1
                    self.total_chunks += len(chunks)
                    
                    if self.verbose:
                        print(f"Added {len(chunks)} chunks from {url} to ChromaDB")
                        
                except Exception as e:
                    print(f"Error adding {url} to ChromaDB: {e}")
            
            # Also yield for potential JSON output
            yield {
                'url': url,
                'text': page_text,
                'chunks_added': len(chunks) if 'chunks' in locals() else 0
            }
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return


def run_scraper(start_url, doc_manager, max_pages=10, verbose=False, output_file=None):
    """
    Run the Scrapy spider with the given parameters.
    
    Args:
        start_url: URL to start scraping from
        doc_manager: DocumentManager instance for ChromaDB integration
        max_pages: Maximum number of pages to scrape
        verbose: Enable verbose output
        output_file: Optional JSON output file for scraped data
    """
    
    # Configure Scrapy settings
    settings = {
        'LOG_LEVEL': 'ERROR' if not verbose else 'INFO',
        'ROBOTSTXT_OBEY': True,  # Respect robots.txt
        'DOWNLOAD_DELAY': 1,     # Be polite to servers
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'CONCURRENT_REQUESTS': 1,  # One request at a time to be respectful
        'USER_AGENT': 'RAG-WebScraper/1.0 (+https://github.com/your-repo)',
    }
    
    # Add JSON output if requested
    if output_file:
        settings['FEEDS'] = {output_file: {'format': 'json'}}
    
    # Create and run crawler
    process = CrawlerProcess(settings=settings)
    
    process.crawl(
        RAGTextSpider,
        start_url=start_url,
        doc_manager=doc_manager,
        max_pages=max_pages,
        verbose=verbose
    )
    
    try:
        process.start()
    except Exception as e:
        print(f"Error during crawling: {e}")
        return False, 0, 0
    
    # Get results from spider
    spider = process.crawlers.pop().spider
    return True, spider.pages_scraped, spider.total_chunks


def get_website_name(url):
    """
    Extract website name from URL for database naming.
    
    Args:
        url: Website URL
        
    Returns:
        Clean website name suitable for database naming
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove common prefixes
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Replace dots and special characters with underscores
        website_name = domain.replace('.', '_').replace('-', '_')
        
        # Remove common suffixes for cleaner names
        for suffix in ['.com', '.org', '.net', '.edu', '.gov', '.co_uk', '.io']:
            suffix_clean = suffix.replace('.', '_')
            if website_name.endswith(suffix_clean):
                website_name = website_name[:-len(suffix_clean)]
                break
        
        # Ensure it's a valid name (alphanumeric + underscore)
        import re
        website_name = re.sub(r'[^a-z0-9_]', '_', website_name)
        
        # Remove multiple underscores
        website_name = re.sub(r'_+', '_', website_name).strip('_')
        
        return website_name if website_name else 'website'
        
    except Exception:
        return 'website'


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scrape websites using Scrapy and add content to ChromaDB for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://example.com
    → Creates database: example_db, collection: example_db
  
  %(prog)s --url https://docs.python.org --max-pages 20
    → Creates database: docs_python_org_db, collection: docs_python_org_db
  
  %(prog)s --url https://blog.example.com --dbname custom_blog
    → Creates database: custom_blog, collection: blog_example_db
  
  %(prog)s --url https://site.com --collection-name custom_collection
    → Creates database: site_db, collection: custom_collection
        """
    )
    
    parser.add_argument(
        '--url', '-u',
        required=True,
        help='URL to start scraping from'
    )
    
    parser.add_argument(
        '--dbname', '-d',
        help='ChromaDB database directory name (default: auto-generated from website)'
    )
    
    parser.add_argument(
        '--collection-name', '-c',
        help='ChromaDB collection name (default: auto-generated from website)'
    )
    
    parser.add_argument(
        '--max-pages', '-m',
        type=int,
        default=10,
        help='Maximum pages to scrape (default: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Optional JSON output file for scraped data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            print(f"Error: Invalid URL: {args.url}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Invalid URL: {args.url} - {e}")
        sys.exit(1)
    
    # Generate website-based names if not provided
    website_name = get_website_name(args.url)
    
    # Set database name
    if args.dbname:
        db_name = args.dbname
    else:
        db_name = f"{website_name}_db"
    
    # Set collection name  
    if args.collection_name:
        collection_name = args.collection_name
    else:
        collection_name = f"{website_name}_db"
    
    # Check if output file already exists
    if args.output and os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists. Remove it or choose a different name.")
        sys.exit(1)
    
    # Initialize DocumentManager
    db_path = f"./{db_name}"
    
    if args.verbose:
        print(f"Website: {website_name}")
        print(f"Database directory: {db_path}")
        print(f"Collection name: {collection_name}")
    
    try:
        doc_manager = DocumentManager(
            chromadb_path=db_path,
            collection_name=collection_name
        )
        
        if doc_manager.chromadb_collection is None:
            print("Error: Failed to initialize ChromaDB. Check database path and permissions.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error initializing DocumentManager: {e}")
        sys.exit(1)
    
    # Run scraper
    print(f"Starting Scrapy crawl from: {args.url}")
    print(f"Website name: {website_name}")
    print(f"Max pages: {args.max_pages}")
    print(f"Database: {db_path}")
    print(f"Collection: {collection_name}")
    
    if args.output:
        print(f"JSON output: {args.output}")
    
    print("-" * 50)
    
    success, pages_scraped, total_chunks = run_scraper(
        start_url=args.url,
        doc_manager=doc_manager,
        max_pages=args.max_pages,
        verbose=args.verbose,
        output_file=args.output
    )
    
    if success:
        # Summary
        print("\n" + "="*50)
        print("SCRAPING COMPLETE")
        print("="*50)
        print(f"Website: {website_name}")
        print(f"Database: {db_path}")
        print(f"Collection: {collection_name}")
        print(f"Pages scraped: {pages_scraped}")
        print(f"Total chunks added: {total_chunks}")
        print(f"Starting URL: {args.url}")
        
        if args.output:
            print(f"JSON output: {args.output}")
        
        print(f"\nContent is now available for RAG queries!")
        print(f"In the UI, select database '{db_name}' and collection '{collection_name}'")
    else:
        print("Scraping failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()