import requests
from bs4 import BeautifulSoup

# Import LLMClient only if you intend to use an LLM
from langchain.llms import LLMClient  # Optional

from langchain.pipelines import Pipeline
# Use MemoryStorage for this example, consider a persistent storage solution for larger projects
from langchain.storage import MemoryStorage

# Replace with your preferred LLM provider and access token (if applicable)
llm = LLMClient(provider="OPENAI", access_token="YOUR_ACCESS_TOKEN")  # Optional

# Function to download and parse HTML content, handling potential errors
def download_and_parse(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for non-200 status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content
        text = soup.get_text(separator='\n')

        # Extract linked URLs (consider filtering based on domain, etc.)
        linked_urls = [link['href'] for link in soup.find_all('a', href=True)
                       if link['href'].startswith('http')]  # Filter for valid URLs
        return text, linked_urls
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None, []  # Handle download errors gracefully

# Define Langchain pipeline
pipeline = Pipeline(
    storage=MemoryStorage(),  # Use a persistent DB for large-scale processing
    stages=[
        # Download and parse the initial document
        lambda ctx: (download_and_parse(ctx.get('url')),),

        # Recursively process linked URLs (consider depth limit or other criteria)
        lambda ctx: [(download_and_parse(url), url) for url in ctx.get('linked_urls', [])],

        # Extract text for each downloaded page
        lambda ctx: [(text, url) for text, url in ctx.flat() if text],  # Filter out empty text

        # Optionally, use LLM to process the text (e.g., summarize)
        # lambda ctx: [(llm.call(text=text), url) for text, url in ctx.flat()],

        # Extract final text content
        lambda ctx: [text for text, _ in ctx.flat()],
    ]
)

# Example usage
url = "https://www.cnn.com/"
results = pipeline.run(url=url)

# Print extracted text for each page (consider further processing)
for text in results:
    print(text)

# For embedding into vector DB
# You can further process the text (e.g., clean, tokenize)
# and use a separate library (e.g., Gensim) to generate embeddings
# and store them in your chosen vector database
