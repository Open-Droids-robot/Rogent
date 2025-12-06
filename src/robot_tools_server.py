from mcp.server.fastmcp import FastMCP
import logging
from duckduckgo_search import DDGS
from googlesearch import search as google_search
import pkgutil
import importlib
import tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_mcp")

# Create the MCP server
mcp = FastMCP("RobotTools")

# Load tools from the tools package
def load_tools(mcp_instance):
    package = tools
    prefix = package.__name__ + "."
    for _, name, _ in pkgutil.iter_modules(package.__path__, prefix):
        try:
            module = importlib.import_module(name)
            if hasattr(module, "register"):
                module.register(mcp_instance)
                logger.info(f"Registered tools from module: {name}")
            else:
                logger.warning(f"Module {name} does not have a register function")
        except Exception as e:
            logger.error(f"Failed to load module {name}: {e}")

load_tools(mcp)


@mcp.tool()
def set_speaking_mode(active: bool) -> str:
    """
    Enable or disable speaking mode (hand gestures).
    
    Args:
        active: True to enable gestures, False to disable.
    """
    state = "ENABLED" if active else "DISABLED"
    logger.info(f"EXECUTING: set_speaking_mode({active})")
    return f"Speaking gestures {state}"

@mcp.tool()
def search_web(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query.
    """
    logger.info(f"EXECUTING: search_web(query='{query}')")
    
    # Try DuckDuckGo (via ddgs package)
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=3)
        if results:
            summary = "Search Results (DuckDuckGo):\n"
            for r in results:
                summary += f"- {r['title']}: {r['body']}\n"
            return summary
    except ImportError:
        logger.warning("ddgs module not found. Falling back to HTML Scraping.")
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}. Falling back to HTML Scraping.")

    # Fallback 1: DuckDuckGo HTML Scraping (Requests + BS4)
    try:
        import requests
        from bs4 import BeautifulSoup
        
        logger.info("Attempting DDG HTML scraping...")
        url = "https://html.duckduckgo.com/html/"
        payload = {'q': query}
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.find_all('div', class_='result'):
            if len(results) >= 3: break
            
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            
            if title_tag and snippet_tag:
                results.append({
                    'title': title_tag.get_text(strip=True),
                    'body': snippet_tag.get_text(strip=True)
                })
        
        if results:
            summary = "Search Results (DDG HTML):\n"
            for r in results:
                summary += f"- {r['title']}: {r['body']}\n"
            return summary
        else:
            logger.warning("DDG HTML scraping returned no results.")

    except Exception as e:
        logger.error(f"DDG HTML scraping failed: {e}")

    # Fallback 2: Google Search
    try:
        # googlesearch-python returns objects with advanced=True
        results = google_search(query, num_results=3, advanced=True)
        summary = "Search Results (Google):\n"
        count = 0
        for r in results:
            summary += f"- {r.title}: {r.description}\n"
            count += 1
            if count >= 3: break
        
        if count > 0:
            return summary
        else:
            return "No results found."
            
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return f"Search failed: {e}"

if __name__ == "__main__":
    # Run the server
    mcp.run()
