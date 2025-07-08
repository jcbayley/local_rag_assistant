import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin, urlparse
import argparse
import os
from w3lib.html import remove_tags, remove_tags_with_content
import re
from readability import Document
from lxml import html

class TextSpider(CrawlSpider):
    name = 'text_spider'
    start_urls = ['https://www.gla.ac.uk/']  # Replace with your URL(s)
    allowed_domains = ['www.gla.ac.uk']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    visited_urls = set()

    def parse_item(self, response):
        url = response.url

        # Skip already visited pages
        if url in self.visited_urls:
            return
    

        self.visited_urls.add(url)

        # Check if content-type is HTML
        content_type = response.headers.get('Content-Type', b'').decode().lower()
        if 'text/html' not in content_type:
            return  # skip non-HTML responses like PDFs, images, etc.

        # Use Readability to parse the main content only
        doc = Document(response.text)
        simplified_html = doc.summary()  # This strips nav/cookie/footer/etc.
        parsed = html.fromstring(simplified_html)
        raw_text = parsed.text_content()

        # Clean up whitespace
        page_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Extract text
        #page_text = ' '.join(response.css('body *::text').getall()).strip()
        print(f"Scraping URL: {url} | Text Length: {len(page_text)}")
        yield {
            'url': url,
            'text': page_text
        }

class TextSpider2(CrawlSpider):
    name = 'text_spider'
    start_urls = ['https://www.gla.ac.uk/']  # Replace with your URL(s)
    allowed_domains = ['www.gla.ac.uk']

    rules = (
        Rule(
            LinkExtractor(
                allow_domains=['www.gla.ac.uk'],
                deny_extensions=[
                    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar', 'exe',
                    'csv', 'ppt', 'pptx', 'jpg', 'jpeg', 'png', 'gif', 'svg', 'mp4', 'mp3'
                ],
                deny=r'(^mailto:|^tel:|\.vcf$|vcard)',
            ),
            callback='parse_item',
            follow=True
        ),
    )

    visited_urls = set()

    def parse_item(self, response):
        url = response.url

        # Skip already visited pages
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        # Parse HTML using lxml
        tree = html.fromstring(response.text)

        # Remove unwanted elements
        for xpath in [
            '//script', '//style', '//nav', '//footer', '//header', '//aside',
            '//*[contains(@class, "cookie")]', '//*[contains(@id, "cookie")]'
        ]:
            for el in tree.xpath(xpath):
                el.drop_tree()

        # Extract visible body text
        text = tree.xpath('//body//text()')
        joined_text = ' '.join(text)

        # Clean whitespace
        page_text = re.sub(r'\s+', ' ', joined_text).strip()

        replace_strings = ["Skip to main content", 
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
                            "Clarity helps us to understand our users\u2019 behaviour by visually representing their clicks, taps and scrolling.",
                            "Switch clarity ON OFF",
                            "Privacy policy close",
                            "Study Research Explore Connect",
                            "Search icon Close menu icon Menu icon bar 1 Menu icon bar 2 Menu icon bar 3"]
        
        for replace_string in replace_strings:
            page_text = page_text.replace(replace_string, "").strip()

        if len(page_text) == 0:
            return
        # Extract text
        #page_text = ' '.join(response.css('body *::text').getall()).strip()
        print(f"Scraping URL: {url} | Text Length: {len(page_text)}")
        yield {
            'url': url,
            'text': page_text
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape text from a website.")
    parser.add_argument('--output', '-o', type=str, default='scrapy_output.json',
                        help="Output file for scraped data in JSON format.")
    args = parser.parse_args()  

    if os.path.exists(args.output):
        raise Exception(f"Output file {args.output} already exists. Please choose a different name or remove the existing file.")
    
    # Crawler settings to output to JSON
    process = CrawlerProcess(settings={
        'FEEDS': {
            f"{args.output}": {'format': 'json'},
        },
        'LOG_LEVEL': 'ERROR'  # Optional: reduce log noise
    })

    process.crawl(TextSpider2)
    process.start()