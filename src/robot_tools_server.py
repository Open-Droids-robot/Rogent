from mcp.server.fastmcp import FastMCP
import logging
import base64
import os
import numpy as np
from PIL import Image
import io
import cv2
import time
from googlesearch import search as google_search
import google.generativeai as genai
from google import genai
from google.genai import types
from dotenv import load_dotenv
from colorama import Fore, Style, init

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

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
    logger.info(f"{Fore.MAGENTA}EXECUTING: get_camera_image{Style.RESET_ALL}")

    # Initialize camera (0 is usually default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open camera")
        return "Error: Could not open camera."

    try:
        # Allow camera to warm up
        for _ in range(15):  # Warm up more frames
            cap.read()

        ret, frame = cap.read()

        if not ret:
            logger.error("Failed to capture frame")
            return "Error: Failed to capture frame."

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Save to outputs folder with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/captured_image_{timestamp}.jpg"
        img.save(filename)
        logger.info(f"Image saved to {filename}")

        # Convert to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    finally:
        cap.release()



@mcp.tool()
@mcp.tool()
def detect_objects() -> str:
    """
    Capture an image and identify objects with normalized coordinates.
    Returns a JSON string: [{"point": [y, x], "label": "name"}, ...]
    """
    logger.info(f"{Fore.MAGENTA}EXECUTING: detect_objects{Style.RESET_ALL}")

    # 1. Capture Image (Reuse logic or call internal helper)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Camera failed"

    try:
        for _ in range(15):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            return "Error: Capture failed"

        # Save for debug
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"outputs/detection_{timestamp}.jpg", frame)

        # Convert for Gemini
        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()

    finally:
        cap.release()

    # 2. Send to Gemini for Processing
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    PROMPT = """
    The label returned should be an identifying name for the object detected.
    The answer should follow the json format: [{"point": <point>,
    "label": <label1>}, ...]. The points are in [y, x] format
    normalized to 0-1000.
    """

    try:
        # Use gemini-2.0-flash-exp for speed/vision or fall back to 1.5-flash
        response = client.models.generate_content(
            model=os.getenv("SCENE_UNDERSTANDING_MODEL"),
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return f"Error analyzing image: {e}"


@mcp.tool()
def move_camera(direction: str) -> str:
    """
    Simulate moving the camera/robot.

    Args:
        direction: Description of where to move (e.g. "left", "right", "pan 30 degrees").
    """
    logger.info(
        f"{Fore.MAGENTA}EXECUTING: move_camera(direction='{direction}'){Style.RESET_ALL}"
    )
    print(
        f"\n{Fore.YELLOW}--- ACTION REQUIRED: PLEASE ROTATE CAMERA MANUALLY ({direction}) ---{Style.RESET_ALL}"
    )
    print(f"{Fore.YELLOW}Waiting 5 seconds for manual adjustment...{Style.RESET_ALL}")
    time.sleep(5)
    print(f"{Fore.GREEN}Resuming...{Style.RESET_ALL}\n")
    return f"Camera moved {direction}. You can now capture an image."




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
    
    # Try DuckDuckGo
    if DDGS:
        try:
            results = DDGS().text(query, max_results=3)
            if results:
                summary = "Search Results (DuckDuckGo):\n"
                for r in results:
                    summary += f"- {r['title']}: {r['body']}\n"
                return summary
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}. Falling back to HTML Scraping.")
    else:
        logger.warning("duckduckgo_search module not found. Falling back to HTML Scraping.")

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
