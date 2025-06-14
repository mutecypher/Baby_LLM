import os
import aiohttp
import asyncio
import logging
import requests
from tqdm.asyncio import tqdm_asyncio  # For async progress bar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path
import re
from gutenberg.cleanup import strip_headers  # From Project Gutenberg Python package

# Setup logging
logging.basicConfig(
    filename='gutenberg_scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories
base_dir = "~/Baby_LLM"
data_dir = os.path.expanduser(f"{base_dir}/data")
gutenberg_dir = os.path.join(data_dir, "gutenberg_b")
cleaned_dir = os.path.join(data_dir, "cleaned_b")
os.makedirs(gutenberg_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

# Range of Gutenberg IDs to download
start_id = 11001
end_id = 30000  # Exclusive, so up to 999

# Base URLs (using aleph.gutenberg.org mirror for reliability)
base_url = "https://gutenberg.org/{path}/{id}/{id}-0.txt"
fallback_url = "https://aleph.gutenberg.org/{path}/{id}/{id}.txt"

# Headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Rate limiting: 1 request every 6 seconds (10 requests per minute)
RATE_LIMIT_DELAY = 6  # Seconds

# Function to compute path for Gutenberg ID (e.g., 1234 -> 1/2/3/4)
def get_gutenberg_path(book_id):
    id_str = str(book_id)
    return "/".join(id_str[:-1]) or "0"

# Function to validate file content
def is_valid_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read(1024)  # Read first 1KB to check
            if not content.strip():
                return False
            # Check for binary or non-text content
            if any(ord(c) > 127 for c in content[:100]) and not re.search(r'[a-zA-Z]', content):
                return False
            return True
    except (UnicodeDecodeError, IOError):
        return False

# Function to clean text (remove headers/footers)
def clean_text(content):
    try:
        cleaned = strip_headers(content).strip()
        # Additional cleaning: remove extra newlines, non-printable characters
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)  # Remove non-ASCII
        return cleaned
    except Exception as e:
        logging.error(f"Cleaning failed: {e}")
        return content

# Async download function with retries
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def download_book(session, book_id, semaphore):
    async with semaphore:
        output_file = Path(gutenberg_dir) / f"{book_id}.txt"
        cleaned_file = Path(cleaned_dir) / f"{book_id}.cleaned.txt"
        
        # Skip if cleaned file exists and is valid
        if cleaned_file.exists() and is_valid_text_file(cleaned_file):
            logging.info(f"Skipping {book_id}: Cleaned file exists")
            return True
        
        # Skip if raw file exists and is valid (but needs cleaning)
        if output_file.exists() and is_valid_text_file(output_file):
            logging.info(f"Processing existing file {book_id}")
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned_content = clean_text(content)
            if cleaned_content:
                with open(cleaned_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                logging.info(f"Saved cleaned file: {book_id}")
                return True
            return False

        # Construct URLs
        path = get_gutenberg_path(book_id)
        urls = [
            base_url.format(path=path, id=book_id),
            fallback_url.format(path=path, id=book_id)
        ]

        for url in urls:
            try:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text(encoding="utf-8", errors="ignore")
                        if len(content) < 1000:  # Arbitrary minimum size
                            logging.warning(f"Book {book_id} too small: {len(content)} bytes")
                            return False
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        cleaned_content = clean_text(content)
                        if cleaned_content:
                            with open(cleaned_file, "w", encoding="utf-8") as f:
                                f.write(cleaned_content)
                            logging.info(f"Downloaded and cleaned book {book_id}")
                            return True
                        else:
                            logging.warning(f"Cleaning failed for book {book_id}")
                            return False
                    else:
                        logging.warning(f"Failed to download book {book_id} from {url}: Status {response.status}")
            except Exception as e:
                logging.error(f"Error downloading {book_id} from {url}: {e}")
        
        logging.error(f"All URLs failed for book {book_id}")
        return False

async def main():
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
    success_count = 0
    failed_count = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        for book_id in range(start_id, end_id):
            tasks.append(download_book(session, book_id, semaphore))
        
        # Run tasks with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading books")
        
        # Count successes and failures
        success_count = sum(1 for r in results if r)
        failed_count = len(results) - success_count

        print(f"Download complete! Files saved to {gutenberg_dir}")
        print(f"Total files downloaded: {success_count}")
        print(f"Total files failed: {failed_count}")
        logging.info(f"Downloaded {success_count} files, failed {failed_count} files")

if __name__ == "__main__":
    print(f"Downloading Gutenberg books with IDs {start_id} to {end_id-1}...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Download interrupted by user")
        logging.info("Download interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        logging.error(f"Error in main loop: {e}")

    # Ensure all files are closed
    print(f"Total files in {gutenberg_dir}: {len(os.listdir(gutenberg_dir))}")
    print(f"Total cleaned files in {cleaned_dir}: {len(os.listdir(cleaned_dir))}")