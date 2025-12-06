from mcp.server.fastmcp import FastMCP
import logging
import base64
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import time
from googlesearch import search as google_search
import glob
import json
import pkgutil
import importlib
import tools
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

# Load tools from the tools package
def load_tools(mcp_instance):
    package = tools
    for importer, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(name)
            if hasattr(module, 'register'):
                module.register(mcp_instance)
                logger.info(f"Loaded tool module: {name}")
            else:
                logger.warning(f"Module {name} does not have a register function")
        except Exception as e:
            logger.error(f"Failed to load module {name}: {e}")

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

@mcp.tool()
def plot_detections(detections_json: str) -> str:
    """
    Visualize detected objects by drawing points/labels on the last captured image.
    
    Args:
        detections_json: A JSON string list of objects.
                         Example: '[{"point": [500, 500], "label": "cup"}]'
                         Points must be [y, x] normalized to 0-1000.
    """
    logger.info(f"EXECUTING: plot_detections")
    
    # 1. Find the most recent image in outputs/
    try:
        list_of_files = glob.glob('outputs/captured_image_*.jpg') 
        if not list_of_files:
            return "Error: No recent image found to plot on."
        
        # Get the latest file created
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Plotting on: {latest_file}")
        
        # 2. Load Image with Pillow
        img = Image.open(latest_file)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Setup Font (try system fonts or default)
        try:
             # Try Mac font first
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except IOError:
            try:
                # Try Linux/Windows common font
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()

        # 3. Parse JSON
        detections = json.loads(detections_json)

        # 4. Plot Logic (from your plot_objects.py)
        point_color = (66, 133, 244) # Google Blue
        text_color = "white"
        outline_color = "white"

        for item in detections:
            # Check for valid point format
            if 'point' not in item: continue
            
            # [y, x] normalized
            norm_y, norm_x = item['point']
            label = item.get('label', 'object')
            
            # Convert to pixels
            x = int((norm_x / 1000) * width)
            y = int((norm_y / 1000) * height)
            
            # Draw Point
            r = 15
            draw.ellipse((x-r, y-r, x+r, y+r), fill=point_color, outline=outline_color, width=4)
            
            # Draw Label
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = x + r + 10
            text_y = y - (text_height / 2)
            
            pad_x, pad_y = 8, 4
            draw.rounded_rectangle(
                (text_x - pad_x, text_y - pad_y, text_x + text_width + pad_x, text_y + text_height + pad_y),
                radius=5, fill=point_color
            )
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # 5. Save Output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = f"outputs/detection_vis_{timestamp}.jpg"
        img.save(outfile)
        logger.info(f"Saved visualization to {outfile}")
        
        return f"Visualization saved to {outfile}"

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return f"Error: {e}"

@mcp.tool()
def plot_trajectory(trajectory_json: str) -> str:
    """
    Visualize a movement trajectory by drawing lines/points on the last captured image.
    
    Args:
        trajectory_json: A JSON string list of steps.
                         Example: '[{"coordinates": [500, 500], "step": "0"}, ...]'
                         Coordinates must be [y, x] normalized to 0-1000.
    """
    logger.info(f"EXECUTING: plot_trajectory")
    
    try:
        # 1. Find the most recent image (or the one we just plotted objects on)
        # We prefer 'detection_vis_' if it exists (so we layer trajectory ON TOP of detections)
        # Otherwise fallback to raw 'captured_image_'
        
        vis_files = glob.glob('outputs/detection_vis_*.jpg')
        raw_files = glob.glob('outputs/captured_image_*.jpg')
        
        target_file = None
        
        # Priority: Latest visualization -> Latest raw capture
        if vis_files:
            target_file = max(vis_files, key=os.path.getctime)
        elif raw_files:
            target_file = max(raw_files, key=os.path.getctime)
            
        if not target_file:
            return "Error: No recent image found to plot trajectory on."
            
        logger.info(f"Plotting trajectory on: {target_file}")
        
        # 2. Load Image
        img = Image.open(target_file)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # 3. Parse JSON
        try:
            steps = json.loads(trajectory_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON string."

        # 4. Draw Logic
        trajectory_color = (52, 168, 83) # Google Green
        outline_color = "white"
        
        pixel_points = []
        
        for item in steps:
            if "coordinates" not in item: continue
            
            # [y, x] normalized
            norm_y, norm_x = item['coordinates']
            
            x = int((norm_x / 1000) * width)
            y = int((norm_y / 1000) * height)
            pixel_points.append((x, y))
            
        # Draw Lines
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=trajectory_color, width=5)
            
        # Draw Points
        for x, y in pixel_points:
            r = 6
            draw.ellipse((x-r, y-r, x+r, y+r), fill=trajectory_color, outline=outline_color, width=2)

        # 5. Save Output
        # We overwrite the target file or create a new "final" one? 
        # Creating a new one preserves history.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = f"outputs/trajectory_vis_{timestamp}.jpg"
        img.save(outfile)
        logger.info(f"Saved trajectory visualization to {outfile}")
        
        return f"Trajectory plotted and saved to {outfile}"

    except Exception as e:
        logger.error(f"Trajectory plotting failed: {e}")
        return f"Error: {e}"

# Load tools from the tools package
load_tools(mcp)

if __name__ == "__main__":
    # Run the server
    mcp.run()
