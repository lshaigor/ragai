import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import deque

class WebCrawler:
    def __init__(self):
        self.visited_pages = set()

    def crawl(self, seed_url, max_pages):
        pages_to_visit = deque([seed_url])
        while pages_to_visit and len(self.visited_pages) < max_pages:
            current_url = pages_to_visit.popleft()
            if current_url not in self.visited_pages:
                self.visited_pages.add(current_url)
                try:
                    response = requests.get(current_url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        self.extract_data(current_url, soup)
                        for link in soup.find_all('a', href=True):
                            absolute_link = self.normalize_url(current_url, link['href'])
                            if absolute_link and absolute_link not in self.visited_pages:
                                pages_to_visit.append(absolute_link)
                except Exception as e:
                    print(f"Failed to crawl {current_url}. Error: {e}")

    def extract_data(self, url, soup):
        title = soup.title.string if soup.title else "No Title Found"
        print(f"Title of {url}: {title}")

    def normalize_url(self, current_url, link):
        parsed_link = urlparse(link)
        if parsed_link.scheme and parsed_link.netloc:  # Absolute URL
            return link
        elif parsed_link.path.startswith(('#', 'mailto:', 'tel:')):  # Ignore non-http links
            return None
        else:  # Relative URL
            base_url = urlparse(current_url)
            return urljoin(current_url, link)

def main():
    crawler = WebCrawler()
    seed_url = 'https://example.com'
    max_pages = 10
    crawler.crawl(seed_url, max_pages)

if __name__ == "__main__":
    main()