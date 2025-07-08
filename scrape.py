import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from urllib.parse import urljoin, urlparse, urldefrag

def is_html(url):
    # Check if the URL ends with an HTML extension or has no extension
    return url.endswith(('.html', '.htm', '.aspx', '.php')) or not urlparse(url).path.split('.')[-1].lower() in ['pdf', 'jpg', 'png', 'gif', 'doc', 'docx']

def normalise_url(url):
    # Remove the fragment from the URL
    return urldefrag(url).url

def scrape_text_and_crawl(root_url, max_pages=10):
    # Set to keep track of visited URLs
    visited = set()
    # List to store the scraped data
    scraped_data = []

    def crawl(url):
        if len(visited) >= max_pages:
            return
        # norn URL
        normalised_url = normalise_url(url)
        # Check if the URL has already been visited
        if normalised_url in visited:
            return
        
        # Check if the URL contains the root URL
        if root_url not in normalised_url:
            return
        
        # Check if the URL points to an HTML file
        if not is_html(normalised_url):
            return
        
        # Mark the URL as visited
        visited.add(normalised_url)
        try:
            # Send a GET request to the URL
            response = requests.get(url, cookies={'cookieconsent_status': 'dismissed'})
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the text from the page
            text = soup.get_text()
            # Store the URL and the scraped text
            scraped_data.append({'url': normalised_url, 'text': text})

            # Find all the links on the page
            for link in soup.find_all('a'):
                # Get the href attribute of the link
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute URLs
                    absolute_url = urljoin(url, href)
                    # Recursively crawl the linked page
                    crawl(absolute_url)
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    crawl(root_url)
    return scraped_data

url = 'https://www.gla.ac.uk/'
scraped_data = scrape_text_and_crawl(url, max_pages=1000)

with open('scraped_data.json', 'w') as f:
    json.dump(scraped_data, f, indent=4)