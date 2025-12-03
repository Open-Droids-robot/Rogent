from mcp.server.fastmcp import FastMCP
import logging
from duckduckgo_search import DDGS
import base64
import os
import numpy as np
from PIL import Image
import io
from googlesearch import search as google_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_mcp")

# Create the MCP server
mcp = FastMCP("RobotTools")

@mcp.tool()
def get_camera_image() -> str:
    """
    Obtain an image from the robot's camera.
    Returns a base64 encoded JPEG string of the image.
    """
    logger.info("EXECUTING: get_camera_image")
    
    # Generate a dummy image (random noise or solid color)
    # In a real scenario, this would capture from cv2.VideoCapture
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

@mcp.tool()
def publish_cmd_vel(linear_x: float, angular_z: float) -> str:
    """
    Publish to cmd_vel to move the robot base.
    
    Args:
        linear_x: Linear velocity (m/s). Positive is forward.
        angular_z: Angular velocity (rad/s). Positive is left.
    """
    logger.info(f"EXECUTING: publish_cmd_vel(linear_x={linear_x}, angular_z={angular_z})")
    return f"Robot moving: linear={linear_x}, angular={angular_z}"

@mcp.tool()
def move_head(pan: float, tilt: float) -> str:
    """
    Move the robot's head.
    
    Args:
        pan: Pan angle in degrees (left/right).
        tilt: Tilt angle in degrees (up/down).
    """
    logger.info(f"EXECUTING: move_head(pan={pan}, tilt={tilt})")
    return f"Head moved to: pan={pan}, tilt={tilt}"

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
