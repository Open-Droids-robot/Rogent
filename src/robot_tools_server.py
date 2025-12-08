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
from google import genai
from google.genai import types

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Configure logging
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Format the message using the standard formatter first
        formatted_message = super().format(record)
        
        # Apply colors based on content
        msg = record.getMessage()
        if "EXECUTING:" in msg:
             return f"{Fore.CYAN}{formatted_message}{Style.RESET_ALL}"
        elif record.levelno >= logging.ERROR:
             return f"{Fore.RED}{formatted_message}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
             return f"{Fore.MAGENTA}{formatted_message}{Style.RESET_ALL}"
             
        return formatted_message

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s:%(name)s:%(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
logger = logging.getLogger("robot_mcp")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

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
    logger.info(f"EXECUTING: get_camera_image")

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
        f"EXECUTING: move_camera(direction='{direction}')"
    )
    logger.info(
        f"{Fore.YELLOW}--- ACTION REQUIRED: PLEASE ROTATE CAMERA MANUALLY ({direction}) ---{Style.RESET_ALL}"
    )
    logger.info(f"{Fore.YELLOW}Waiting 5 seconds for manual adjustment...{Style.RESET_ALL}")
    time.sleep(5)
    logger.info(f"{Fore.GREEN}Resuming...{Style.RESET_ALL}")
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

@mcp.tool()
def plot_bounding_boxes(detections_json: str) -> str:
    """
    Visualize detected objects by drawing bounding boxes/labels on the last captured image.
    
    Args:
        detections_json: A JSON string list of objects.
                         Example: '[{"box_2d": [ymin, xmin, ymax, xmax], "label": "cup"}]'
                         Coordinates must be normalized to 0-1000.
    """
    logger.info(f"EXECUTING: plot_bounding_boxes")
    
    try:
        # 1. Find the most recent image in outputs/
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
        
        # Setup Font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()

        # 3. Parse JSON
        detections = json.loads(detections_json)
        
        # Define a list of nice colors to cycle through
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
            (128, 0, 128),  # Purple
        ]

        # 4. Plot Logic
        for i, item in enumerate(detections):
            if 'box_2d' not in item: continue
            
            # [ymin, xmin, ymax, xmax] normalized
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = item['box_2d']
            label = item.get('label', 'object')
            
            # Pick a color
            color = colors[i % len(colors)]
            
            # Convert to pixels
            ymin = int((ymin_norm / 1000) * height)
            xmin = int((xmin_norm / 1000) * width)
            ymax = int((ymax_norm / 1000) * height)
            xmax = int((xmax_norm / 1000) * width)
            
            # Draw Rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
            
            # Draw Label
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text above the box if possible, else inside top
            text_x = xmin
            text_y = ymin - text_height - 6
            if text_y < 0:
                text_y = ymin + 4
            
            pad = 4
            draw.rectangle(
                (text_x - pad, text_y - pad, text_x + text_width + pad, text_y + text_height + pad),
                fill=color
            )
            draw.text((text_x, text_y), label, fill="white", font=font)

        # 5. Save Output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = f"outputs/bounded_boxes_{timestamp}.jpg"
        img.save(outfile)
        logger.info(f"Saved visualization to {outfile}")
        
        return f"Visualization saved to {outfile}"

    except Exception as e:
        logger.error(f"Bounding box plotting failed: {e}")
        return f"Error: {e}"

@mcp.tool()
def analyze_scene(instruction: str) -> str:
    """
    Analyzes the current scene using the robot's camera to perform object detection, 
    scene understanding, or trajectory planning.
    
    Args:
        instruction: The specific task (e.g. "Find the red cup", "Describe the scene", "Plan a path to the door").
    """
    logger.info(f"EXECUTING: analyze_scene(instruction='{instruction}')")
    
    # 1. Capture Image (reuse existing tool logic)
    img_b64 = get_camera_image()
    if img_b64.startswith("Error"):
        return img_b64
        
    try:
        # 2. Setup Client
        if not GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY not found in environment."
            
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Use a fast, vision-capable model for the inner loop
        model_name = os.getenv('PERCEPTION_MODEL', 'gemini-2.5-flash')
        
        # 3. Define the Perception System Prompt
        perception_prompt = """
        You are a Scene Understanding AI for a robot.
        Analyze the provided image based on the User's Instruction.
        
        OUTPUT FORMATS:
        
        A) FOR OBJECT DETECTION ("Find the...", "Where is...", "Point to..."):
           Return a JSON list: [{"point": [y, x], "label": "object_name"}]
           - y, x are normalized coordinates (0-1000).
           
        B) FOR BOUNDING BOX DETECTION ("Detect objects with boxes", "Box the...", "Draw boxes"):
           Return bounding boxes as a JSON array with labels. Never return masks
           or code fencing. Limit to 25 objects. Include as many objects as you
           can identify on the table.

           If an object is present multiple times, name them according to their
           unique characteristic (colors, size, position, unique characteristics, etc..).

           The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax],
           "label": <label for the object>}] normalized to 0-1000. The values in
           box_2d must only be integers.
           
        C) FOR TRAJECTORY PLANNING ("Plan a path...", "Go to..."):
           Return a JSON list: [{"coordinates": [y, x], "step": "step_number"}]
           - y, x are normalized coordinates (0-1000).
           
        D) FOR DESCRIPTION ("Describe...", "What do you see?"):
           Return a natural language description.
           
        CRITICAL RULES:
        - If returning JSON, output ONLY the raw JSON string. Do not use Markdown code blocks.
        - Normalize coordinates: [0,0] is top-left, [1000,1000] is bottom-right.
        - If the user asks for a path/trajectory, provide at least 3-5 waypoints.
        """
        
        # 4. Call the Model
        img_bytes = base64.b64decode(img_b64)
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=perception_prompt),
                        types.Part(text=f"User Instruction: {instruction}"),
                        types.Part(inline_data=types.Blob(mime_type='image/jpeg', data=img_bytes))
                    ]
                )
            ]
        )
        
        if not response.text:
            return "Vision model returned no text."
            
        result_text = response.text.strip()
        
        # 5. Handle Result & Plotting
        # Clean up any potential markdown formatting
        cleaned_text = result_text.replace('```json', '').replace('```', '').strip()
        
        is_json = False
        parsed_data = None
        try:
            parsed_data = json.loads(cleaned_text)
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                is_json = True
        except:
            pass
            
        if is_json and parsed_data:
            first_item = parsed_data[0]
            
            # Case A: Object Detection
            if "point" in first_item:
                # We call the plot_detections tool directly
                plot_msg = plot_detections(cleaned_text)
                return f"Objects Detected: {cleaned_text}\n{plot_msg}"
            
            # Case A.2: Bounding Boxes
            elif "box_2d" in first_item:
                plot_msg = plot_bounding_boxes(cleaned_text)
                return f"Objects Detected (Boxes): {cleaned_text}\n{plot_msg}"
                
            # Case B: Trajectory
            elif "coordinates" in first_item:
                # We call the plot_trajectory tool directly
                plot_msg = plot_trajectory(cleaned_text)
                return f"Trajectory Planned: {cleaned_text}\n{plot_msg}"
        
        # Case C: Description or text response
        return f"Scene Analysis: {result_text}"

    except Exception as e:
        logger.error(f"Analyze Scene Error: {e}")
        return f"Error analyzing scene: {e}"

# Load tools from the tools package
load_tools(mcp)

if __name__ == "__main__":
    # Run the server
    mcp.run()
