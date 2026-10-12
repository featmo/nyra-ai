import datetime
from ddgs import DDGS
import psutil
import os

def get_date_time() -> str:
    """Returns the current date and time in a human-readable format."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def web_search(query: str) -> str:
    """Search the web for current information, news and facts. Returns titles,
    links and short snippets. Use when you need up-to-date or unfamiliar information.

    Args:
        query: What to search for.
    """
    try:
        print(f"Searching the web for you")
        results = DDGS().text(query, safesearch="off", max_results=5)
        if not results:
            return "No results found."
        return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)
    except Exception as e:
        return f"Search failed: {e}"

def web_fetch(url: str) -> str:
    """Fetch a web page and return its main text. Use after web_search to read
    one of the results in full.

    Args:
        url: The full URL of the page.
    """
    try:
        print(f"Fetching page: {url}")
        page = DDGS().extract(url, fmt="text_plain")
        return page["content"][:4000]
    except Exception as e:
        return f"Could not read page: {e}"

def save_to_file(filename: str, content: str) -> str:
    """Save content to a file.

    Args:
        filename: The name of the file to save to.
        content: The content to save.
    """
    # get current directory
    current_dir = os.getcwd()
    try:
        with open(os.path.join(current_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
        return f"Content saved to {filename}"
    except Exception as e:
        return f"Could not save to file: {e}"

# list directory contents platform agnostic
def list_directory(path: str = ".") -> str:
    """List the contents of a directory.

    Args:
        path: The path to the directory to list.
        if not specified, lists the current directory.
        get path type
    """
    try:
        # get path and path type
        contents = os.listdir(path)
        return "\n".join(contents)
    except Exception as e:
        return f"Could not list directory: {e}"

def get_current_directory() -> str:
    """Returns the current working directory."""
    return os.getcwd()

def get_file_contents(filename: str) -> str:
    """Returns the contents of a file.

    Args:
        filename: The name of the file to read.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Could not read file: {e}"