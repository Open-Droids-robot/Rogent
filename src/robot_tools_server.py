from mcp.server.fastmcp import FastMCP
import logging
import base64
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import time
import re
import weave
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


def get_output_path(filename: str, session_id: str = None) -> str:
    """Helper to determine file path based on session_id."""
    base_folder = "outputs"
    if session_id:
        # Sanitize session_id just in case
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ("-", "_")])
        target_folder = os.path.join(base_folder, safe_id)
    else:
        # Default to date folder if no session provided
        date_str = time.strftime("%Y-%m-%d")
        target_folder = os.path.join(base_folder, date_str)

    os.makedirs(target_folder, exist_ok=True)
    return os.path.join(target_folder, filename)


def _get_latest_captured_image_path(session_id: str = None) -> str:
    """
    Helper to find the most recent captured image file path.
    Returns the file path or raises an exception if not found.
    """
    if session_id:
        base_search = os.path.join("outputs", session_id)
        list_of_files = glob.glob(os.path.join(base_search, "captured_image_*.jpg"))
    else:
        list_of_files = glob.glob("outputs/**/captured_image_*.jpg", recursive=True)

    if not list_of_files:
        raise FileNotFoundError("No recent image found")

    return max(list_of_files, key=os.path.getctime)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Format the message using the standard formatter first
        formatted_message = super().format(record)

        # Apply colors based on content
        msg = record.getMessage()
        if "EXECUTING:" in msg:
            return f"{Fore.CYAN}{formatted_message}{Style.RESET_ALL}"
        elif "SUCCESS:" in msg or "SAVED:" in msg:
            return f"{Fore.GREEN}{formatted_message}{Style.RESET_ALL}"
        elif "DETECTED:" in msg:
            return f"{Fore.CYAN}{formatted_message}{Style.RESET_ALL}"
        elif record.levelno >= logging.ERROR:
            return f"{Fore.RED}{formatted_message}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            return f"{Fore.MAGENTA}{formatted_message}{Style.RESET_ALL}"

        return formatted_message


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(levelname)s:%(name)s:%(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
logger = logging.getLogger("robot_mcp")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Weave
try:
    weave.init(os.getenv("WEAVE_PROJECT_ID", "open-droids/test-agent-v2"))
except Exception as e:
    logger.warning(f"Weave init failed: {e}")

# Create the MCP server
mcp = FastMCP("RobotTools")


# Load tools from the tools package
def load_tools(mcp_instance):
    package = tools
    for importer, name, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        try:
            module = importlib.import_module(name)
            if hasattr(module, "register"):
                module.register(mcp_instance)
                logger.info(f"Loaded tool module: {name}")
            else:
                logger.warning(f"Module {name} does not have a register function")
        except Exception as e:
            logger.error(f"Failed to load module {name}: {e}")


@mcp.tool()
def get_latest_image(session_id: str = None) -> str:
    """
    Retrieve the most recently captured image from storage (does NOT take a new photo).
    Returns a base64 encoded JPEG string of the cached image.
    """
    logger.info(f"EXECUTING: get_latest_image (from cache)")

    try:
        latest_file = _get_latest_captured_image_path(session_id)
        logger.info(f"Retrieved cached image: {latest_file}")

        img = Image.open(latest_file)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str
    except FileNotFoundError:
        return "Error: No recent image found in cache."
    except Exception as e:
        logger.error(f"Failed to retrieve cached image: {e}")
        return f"Error: {e}"


# TODO: Rename get_camera_image
@mcp.tool()
def get_camera_image(camera_type: str = "head", session_id: str = None) -> str:
    """
    Obtain an image from one of the robot's cameras.

    Args:
        camera_type: Type of camera to use. Options: "head", "left_wrist", "right_wrist"
        session_id: Optional session identifier for organizing outputs

    Returns a base64 encoded JPEG string of the image.
    """
    # Map camera types to video device indices
    camera_map = {
        "head": 0,  # ZED 2 camera
        "left_wrist": 2,  # RealSense
        "right_wrist": 6,  # RealSense
    }

    if camera_type not in camera_map:
        error_msg = f"Invalid camera_type: '{camera_type}'. Valid options: {list(camera_map.keys())}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    camera_index = camera_map[camera_type]
    logger.info(
        f"EXECUTING: get_camera_image(camera_type='{camera_type}', index={camera_index})"
    )

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(f"Could not open {camera_type} camera (index {camera_index})")
        return f"Error: Could not open {camera_type} camera."

    try:
        # Allow camera to warm up
        for _ in range(15):  # Warm up more frames
            cap.read()

        ret, frame = cap.read()

        if not ret:
            logger.error(f"Failed to capture frame from {camera_type} camera")
            return f"Error: Failed to capture frame from {camera_type} camera."

        # Handle ZED Camera (Side-by-Side Stereo) - only for head camera
        height, width, _ = frame.shape
        if (
            camera_type == "head" and width > height * 1.8
        ):  # Simple heuristic for side-by-side
            # Crop to get just the left eye (first half of width)
            frame = frame[:, : width // 2, :]
            logger.info("DETECTED: Stereo image - Cropped to left eye.")

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Optimization: Resize to max 800px width if larger
        if img.size[0] > 800:
            new_height = int(800 * img.size[1] / img.size[0])
            img = img.resize((800, new_height), Image.Resampling.LANCZOS)
            logger.info(f"DETECTED: Image too large, resized to 800x{new_height}")

        # Save to outputs folder with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = get_output_path(
            f"captured_image_{camera_type}_{timestamp}.jpg", session_id
        )
        img.save(filename)
        logger.info(f"SAVED: Image from {camera_type} camera saved to {filename}")
        # Save camera/head pose metadata for this captured frame so object coordinates
        # can later be interpreted relative to the camera frame.
        try:
            camera_pos = _get_current_camera_position()
            meta = {
                "image_path": filename,
                "camera_type": camera_type,
                "timestamp": timestamp,
                "camera_position": camera_pos,
                "image_size": {"width": img.size[0], "height": img.size[1]},
            }
            meta_filename = f"{filename}.meta.json"
            with open(meta_filename, "w") as mf:
                json.dump(meta, mf)
            logger.info(f"SAVED: Camera metadata to {meta_filename}")
        except Exception as e:
            logger.warning(f"Failed to save camera metadata for {filename}: {e}")

        # Convert to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    finally:
        cap.release()


# Global state to track camera position (simulated for now)
# In a real robot, this would query the hardware.
_CAMERA_STATE = {"pan": 500, "tilt": 500}


def _get_current_camera_position():
    """Helper to get current pan/tilt values."""
    return _CAMERA_STATE


# @mcp.tool()
# def move_camera(direction: str, pan: int = None, tilt: int = None) -> str:
#     """
#     Simulate moving the camera/robot using servo commands.

#     Args:
#         direction: Description of where to move (e.g. "left", "right", "up", "down").
#         pan: (Optional) Exact servo value for panning (ID 2). Range 1 (right) to 1000 (left).
#         tilt: (Optional) Exact servo value for tilting (ID 1). Range 1 (down) to 1000 (up).
#     """
#     logger.info(
#         f"EXECUTING: move_camera(direction='{direction}', pan={pan}, tilt={tilt})"
#     )

#     # Get current state
#     current_state = _get_current_camera_position()
#     current_pan = current_state["pan"]
#     current_tilt = current_state["tilt"]
#     step_size = 200

#     # Calculate target values if direction string is used
#     target_pan = pan
#     target_tilt = tilt

#     if target_pan is None and target_tilt is None:
#         direction_lower = direction.lower()
#         if "left" in direction_lower:
#             target_pan = current_pan + step_size  # Left is higher value (towards 1000)
#         elif "right" in direction_lower:
#             target_pan = current_pan - step_size  # Right is lower value (towards 1)
#         else:
#             target_pan = current_pan  # Keep current if not changing

#         if "up" in direction_lower:
#             target_tilt = current_tilt + step_size  # Up is higher value (towards 1000)
#         elif "down" in direction_lower:
#             target_tilt = current_tilt - step_size  # Down is lower value (towards 1)
#         else:
#             target_tilt = current_tilt  # Keep current if not changing

#         # Clamp values
#         if target_pan is not None:
#             target_pan = max(1, min(1000, target_pan))
#         if target_tilt is not None:
#             target_tilt = max(1, min(1000, target_tilt))

#     # Update global state (Simulated hardware update)
#     if target_pan is not None:
#         _CAMERA_STATE["pan"] = target_pan
#     if target_tilt is not None:
#         _CAMERA_STATE["tilt"] = target_tilt

#     # Construct the command log
#     cmd_log = []
#     if target_pan is not None:
#         cmd_log.append(f"Servo ID 2 (Pan): {target_pan} [1=Right, 1000=Left]")
#     if target_tilt is not None:
#         cmd_log.append(f"Servo ID 1 (Tilt): {target_tilt} [1=Down, 1000=Up]")

#     logger.info(
#         f"{Fore.YELLOW}--- ACTION REQUIRED: PLEASE ROTATE CAMERA MANUALLY ({direction}) ---{Style.RESET_ALL}"
#     )
#     if cmd_log:
#         for log in cmd_log:
#             logger.info(f"{Fore.YELLOW}  -> {log}{Style.RESET_ALL}")

#     logger.info(
#         f"{Fore.YELLOW}Waiting 5 seconds for manual adjustment...{Style.RESET_ALL}"
#     )
#     time.sleep(5)
#     logger.info(f"{Fore.GREEN}Resuming...{Style.RESET_ALL}")

#     return f"Camera moved {direction}. New targets: Pan={_CAMERA_STATE['pan']}, Tilt={_CAMERA_STATE['tilt']}."


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
            logger.warning(
                f"DuckDuckGo search failed: {e}. Falling back to HTML Scraping."
            )
    else:
        logger.warning(
            "duckduckgo_search module not found. Falling back to HTML Scraping."
        )

    # Fallback 1: DuckDuckGo HTML Scraping (Requests + BS4)
    try:
        import requests
        from bs4 import BeautifulSoup

        logger.info("Attempting DDG HTML scraping...")
        url = "https://html.duckduckgo.com/html/"
        payload = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.find_all("div", class_="result"):
            if len(results) >= 3:
                break

            title_tag = result.find("a", class_="result__a")
            snippet_tag = result.find("a", class_="result__snippet")

            if title_tag and snippet_tag:
                results.append(
                    {
                        "title": title_tag.get_text(strip=True),
                        "body": snippet_tag.get_text(strip=True),
                    }
                )

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
            if count >= 3:
                break

        if count > 0:
            return summary
        else:
            return "No results found."

    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return f"Search failed: {e}"


@mcp.tool()
def plot_detections(detections_json: str, session_id: str = None) -> str:
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
        try:
            latest_file = _get_latest_captured_image_path(session_id)
        except FileNotFoundError:
            return "Error: No recent image found to plot on."

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
        point_color = (66, 133, 244)  # Google Blue
        text_color = "white"
        outline_color = "white"

        for item in detections:
            # Check for valid point format
            if "point" not in item:
                continue

            # [y, x] normalized
            norm_y, norm_x = item["point"]
            label = item.get("label", "object")

            # Convert to pixels
            x = int((norm_x / 1000) * width)
            y = int((norm_y / 1000) * height)

            # Draw Point
            r = 15
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=point_color,
                outline=outline_color,
                width=4,
            )

            # Draw Label
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text above the box if possible, else inside top
            text_x = x + r + 10
            text_y = y - (text_height / 2)

            pad_x, pad_y = 8, 4
            draw.rounded_rectangle(
                (
                    text_x - pad_x,
                    text_y - pad_y,
                    text_x + text_width + pad_x,
                    text_y + text_height + pad_y,
                ),
                radius=5,
                fill=point_color,
            )
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        # 5. Save Output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = get_output_path(f"detection_vis_{timestamp}.jpg", session_id)
        img.save(outfile)
        logger.info(f"SAVED: Visualization to {outfile}")

        return f"Visualization saved to {outfile}"

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return f"Error: {e}"


@mcp.tool()
def plot_trajectory(trajectory_json: str, session_id: str = None) -> str:
    """
    Visualize a movement trajectory by drawing lines/points on the last captured image.

    Args:
        trajectory_json: A JSON string list of steps.
                         Example: '[{"coordinates": [500, 500], "step": "0"}, ...]'
                         Coordinates must be [y, x] normalized to 0-1000.
    """
    logger.info(f"EXECUTING: plot_trajectory")

    try:
        # 1. Find the most recent image
        # Priority: Latest visualization -> Latest raw capture
        # NOTE: Using raw capture via helper for simplicity and consistency with other tools.
        # If we specifically want to draw on TOP of detections, we'd need extra logic.
        # But usually we just draw on the base image.
        try:
            target_file = _get_latest_captured_image_path(session_id)
        except FileNotFoundError:
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
        trajectory_color = (52, 168, 83)  # Google Green
        outline_color = "white"

        pixel_points = []

        for item in steps:
            if "coordinates" not in item:
                continue

            # [y, x] normalized
            norm_y, norm_x = item["coordinates"]

            x = int((norm_x / 1000) * width)
            y = int((norm_y / 1000) * height)
            pixel_points.append((x, y))

        # Draw Lines
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=trajectory_color, width=5)

        # Draw Points
        for x, y in pixel_points:
            r = 6
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=trajectory_color,
                outline=outline_color,
                width=2,
            )

        # 5. Save Output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = get_output_path(f"trajectory_vis_{timestamp}.jpg", session_id)
        img.save(outfile)
        logger.info(f"SAVED: Trajectory visualization to {outfile}")

        return f"Trajectory plotted and saved to {outfile}"

    except Exception as e:
        logger.error(f"Trajectory plotting failed: {e}")
        return f"Error: {e}"


@mcp.tool()
def plot_bounding_boxes(detections_json: str, session_id: str = None) -> str:
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
        try:
            latest_file = _get_latest_captured_image_path(session_id)
        except FileNotFoundError:
            return "Error: No recent image found to plot on."

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
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
            (128, 0, 128),  # Purple
        ]

        # 4. Plot Logic
        for i, item in enumerate(detections):
            if "box_2d" not in item:
                continue

            # [ymin, xmin, ymax, xmax] normalized
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = item["box_2d"]
            label = item.get("label", "object")

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
                (
                    text_x - pad,
                    text_y - pad,
                    text_x + text_width + pad,
                    text_y + text_height + pad,
                ),
                fill=color,
            )
            draw.text((text_x, text_y), label, fill="white", font=font)

        # 5. Save Output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = get_output_path(f"bounded_boxes_{timestamp}.jpg", session_id)
        img.save(outfile)
        logger.info(f"SAVED: Visualization to {outfile}")

        return f"Visualization saved to {outfile}"

    except Exception as e:
        logger.error(f"Bounding box plotting failed: {e}")
        return f"Error: {e}"


@mcp.tool()
def denormalize_coords(x_norm: float, y_norm: float, session_id: str = None) -> str:
    """
    Convert normalized coordinates (0-1000) back to pixel or raw camera coordinates.

    Args:
        x_norm: Normalized x coordinate (0-1000).
        y_norm: Normalized y coordinate (0-1000).
    """
    logger.info(f"EXECUTING: denormalize_coords(x={x_norm}, y={y_norm})")

    # 1. Get image dimensions from latest image
    try:
        try:
            latest_file = _get_latest_captured_image_path(session_id)
        except FileNotFoundError:
            return "Error: No recent image found to determine resolution."

        img = Image.open(latest_file)
        width, height = img.size
    except Exception as e:
        logger.error(f"Failed to load latest image dimensions: {e}")
        return f"Error: {e}"

    # 2. Denormalize
    try:
        x_val = float(x_norm)
        y_val = float(y_norm)
    except ValueError:
        return "Error: Coordinates must be numbers."

    x_pixel = int((x_val / 1000) * width)
    y_pixel = int((y_val / 1000) * height)

    result = {
        "pixel_coords": [x_pixel, y_pixel],
        "image_dims": [width, height],
        "normalized_input": [x_val, y_val],
    }

    logger.info(f"Denormalized coordinates: {result}")

    return json.dumps(result)


# --- HELPER FOR VISION CALLS ---
def _call_vision_model(
    prompt: str, instruction: str, image_b64: str, json_mode: bool = False
) -> str:
    """Internal helper to call the vision model."""
    try:
        if not GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY not found in environment."

        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Default to the same model as the agent or a known capable vision model
        model_name = os.getenv("PERCEPTION_MODEL", "gemini-2.0-flash-exp")
        logger.info(f"Using vision model: {model_name} (JSON mode: {json_mode})")

        img_bytes = base64.b64decode(image_b64)

        config = None
        if json_mode:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=os.getenv("TEMPERATURE", 0.5),
                thinking_config=types.ThinkingConfig(
                    thinking_budget=os.getenv("THINKING_BUDGET", 0)
                ),
            )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(text=f"User Instruction: {instruction}"),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg", data=img_bytes
                            )
                        ),
                    ],
                )
            ],
            config=config,
        )

        if not response.text:
            return "Error: Vision model returned no text."

        return response.text.strip()
    except Exception as e:
        logger.error(f"Vision Call Error: {e}")
        return f"Error calling vision model: {e}"


@mcp.tool()
def understand_scene(
    instruction: str,
    use_cached_image: bool = False,
    camera_type: str = "head",
    session_id: str = None,
) -> str:
    """
    Get a natural language description of the scene.
    Args:
        instruction: Specific question (e.g. "Describe the scene", "Is it safe to move?").
        use_cached_image: If True, use the most recent captured image instead of taking a new one.
        camera_type: Camera to use if not cached. Options: "head", "left_wrist", "right_wrist".
    """
    logger.info(
        f"EXECUTING: understand_scene('{instruction}', cached={use_cached_image}, camera={camera_type})"
    )

    if use_cached_image:
        img_b64 = get_latest_image(session_id)
    else:
        img_b64 = get_camera_image(camera_type, session_id)

    if img_b64.startswith("Error"):
        return img_b64

    prompt = """
    You are a Scene Understanding AI for a robot.
    Provide a clear, natural language description of the scene based on the user's instruction.
    Do NOT return JSON. Just text.
    """
    return _call_vision_model(prompt, instruction, img_b64)


@mcp.tool()
def detect_objects(
    instruction: str,
    use_cached_image: bool = False,
    camera_type: str = "head",
    session_id: str = None,
) -> str:
    """
    Detect objects and return their point locations.
    Args:
        instruction: E.g. "Find the cup", "Where is the bottle?".
        use_cached_image: If True, use the most recent captured image instead of taking a new one.
        camera_type: Camera to use if not cached. Options: "head", "left_wrist", "right_wrist".
    """
    logger.info(
        f"EXECUTING: detect_objects('{instruction}', cached={use_cached_image}, camera={camera_type})"
    )

    if use_cached_image:
        img_b64 = get_latest_image(session_id)
    else:
        img_b64 = get_camera_image(camera_type, session_id)

    if img_b64.startswith("Error"):
        return img_b64

    prompt = """
    You are an Object Detection AI.
    Return a JSON list of objects: [{"point": [y, x], "label": "object_name"}]
    - y, x are normalized coordinates (0-1000).
    - [0,0] is top-left.
    - If the requested object is not clearly visible, return an empty list [].
    - NEVER hallucinate objects. Only report what is clearly visible.
    - Output ONLY raw JSON. No markdown.
    """

    result = _call_vision_model(prompt, instruction, img_b64, json_mode=True)
    if result.startswith("Error"):
        return result

    cleaned_text = result.replace("```json", "").replace("```", "").strip()
    # Save detections + camera metadata so each detection can be traced to the camera frame.
    try:
        detections = json.loads(cleaned_text)
    except Exception:
        detections = None

    try:
        # Use the most recently captured image as the reference
        latest_image = _get_latest_captured_image_path(session_id)
    except Exception:
        latest_image = None

    try:
        # Prefer metadata saved at capture time, otherwise query current camera state
        camera_meta = None
        if latest_image:
            meta_path = f"{latest_image}.meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as mf:
                    camera_meta = json.load(mf)
        if camera_meta is None:
            camera_meta = {"camera_position": _get_current_camera_position()}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        detections_meta = {
            "image_path": latest_image,
            "timestamp": timestamp,
            "camera_meta": camera_meta,
            "detections": detections,
            "raw_detection_text": cleaned_text,
        }
        meta_outfile = get_output_path(f"detections_meta_{timestamp}.json", session_id)
        with open(meta_outfile, "w") as outf:
            json.dump(detections_meta, outf, indent=2)
        logger.info(f"SAVED: Detection metadata to {meta_outfile}")
    except Exception as e:
        logger.warning(f"Failed to save detection metadata: {e}")

    plot_msg = plot_detections(cleaned_text, session_id)
    return f"Detections: {cleaned_text}\n{plot_msg}\nMetadata saved to {meta_outfile if 'meta_outfile' in locals() else 'n/a'}"


@mcp.tool()
def get_bounded_boxes(
    instruction: str,
    use_cached_image: bool = False,
    camera_type: str = "head",
    session_id: str = None,
) -> str:
    """
    Detect objects and return/plot bounding boxes.
    Args:
        instruction: E.g. "Box all the fruits", "Draw a box around the cup".
        use_cached_image: If True, use the most recent captured image instead of taking a new one.
        camera_type: Camera to use if not cached. Options: "head", "left_wrist", "right_wrist".
    """
    logger.info(
        f"EXECUTING: get_bounded_boxes('{instruction}', cached={use_cached_image}, camera={camera_type})"
    )

    if use_cached_image:
        img_b64 = get_latest_image(session_id)
    else:
        img_b64 = get_camera_image(camera_type, session_id)

    if img_b64.startswith("Error"):
        return img_b64

    prompt = """
    You are an Object Detection AI using Bounding Boxes.
    Return a JSON list: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "name"}]
    - Coordinates normalized to 0-1000.
    - Output ONLY raw JSON. No markdown.
    - Limit to 25 objects.
    """

    result = _call_vision_model(prompt, instruction, img_b64, json_mode=True)
    if result.startswith("Error"):
        return result

    cleaned_text = result.replace("```json", "").replace("```", "").strip()
    plot_msg = plot_bounding_boxes(cleaned_text, session_id)
    return f"Boxes: {cleaned_text}\n{plot_msg}"


@mcp.tool()
def get_trajectory(
    instruction: str,
    use_cached_image: bool = False,
    camera_type: str = "head",
    session_id: str = None,
) -> str:
    """
    Plan a movement trajectory.
    Args:
        instruction: E.g. "Plan a path to the door".
        use_cached_image: If True, use the most recent captured image instead of taking a new one.
        camera_type: Camera to use if not cached. Options: "head", "left_wrist", "right_wrist".
    """
    logger.info(
        f"EXECUTING: get_trajectory('{instruction}', cached={use_cached_image}, camera={camera_type})"
    )

    if use_cached_image:
        img_b64 = get_latest_image(session_id)
    else:
        img_b64 = get_camera_image(camera_type, session_id)

    if img_b64.startswith("Error"):
        return img_b64

    prompt = """
    You are a Trajectory Planning AI.
    Return a JSON list of waypoints: [{"coordinates": [y, x], "step": "0"}]
    - Coordinates normalized to 0-1000.
    - Provide at least 3-5 waypoints.
    - Output ONLY raw JSON. No markdown.
    """

    result = _call_vision_model(prompt, instruction, img_b64, json_mode=True)
    if result.startswith("Error"):
        return result

    cleaned_text = result.replace("```json", "").replace("```", "").strip()
    plot_msg = plot_trajectory(cleaned_text, session_id)
    return f"Trajectory: {cleaned_text}\n{plot_msg}"


@weave.op()
def log_search_step(step: int, image_path: str, result: str):
    """Log a search step with image to Weave."""
    try:
        img = Image.open(image_path)
        return {"step": step, "image": img, "result": result}
    except Exception as e:
        logger.error(f"Failed to log step to weave: {e}")
        return str(e)


def semantic_match_target(target_object: str, detected_labels: list[str]) -> dict:
    """Uses LLM to semantically match target against detected labels."""
    if not detected_labels:
        return {"found": False}
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt = f"""
        Target: "{target_object}"
        Candidates: {json.dumps(detected_labels)}
        Task: Return JSON {{"found": bool, "match": "str"}} if any candidate matches target semantically.
        Rules: Conservative match. Ignore irrelevant adjectives. No hallucination.
        """
        response = client.models.generate_content(
            model=os.getenv("PERCEPTION_MODEL", "gemini-2.0-flash-exp"),
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"Semantic match error: {e}")
        return {"found": False}


@mcp.tool()
@weave.op()
def search_until_found(
    object_name: str,
    max_attempts: int = 8,
    session_id: str = None,
) -> str:
    """
    Loops and searches for an object by moving the camera until it is found.
    The next move direction is decided dynamically by the vision model if the object is missing.

    Args:
        object_name: The object to search for (e.g. "red bottle").
        max_attempts: Maximum number of search steps (default: 5).
        session_id: Optional session identifier.
    """
    logger.info(f"EXECUTING: search_until_found(object='{object_name}')")

    for i in range(max_attempts):
        logger.info(f"{Fore.YELLOW}Search Step {i + 1}/{max_attempts}{Style.RESET_ALL}")

        # 1. Detect objects (Object-agnostic)
        instruction = "Detect all visible objects in the scene"
        # Always force a new image capture
        detection_result = detect_objects(
            instruction=instruction,
            use_cached_image=False,
            session_id=session_id,
        )

        # Log to Weave if visualization exists
        try:
            # detection_result often ends with "Visualization saved to path/to/file.jpg"
            match = re.search(r"Visualization saved to (.*)", detection_result)
            if match:
                vis_path = match.group(1).strip()
                log_search_step(i + 1, vis_path, detection_result)
            else:
                # If no visualization msg, try to get latest raw image
                latest_raw = _get_latest_captured_image_path(session_id)
                log_search_step(i + 1, latest_raw, detection_result)
        except Exception as e:
            logger.warning(f"Could not log step to Weave: {e}")

        # 2. Check if found
        detections = []
        if "Detections:" in detection_result:
            try:
                # Extract JSON list part
                json_part = detection_result.split("Detections:", 1)[1].strip()
                # Find the end of the JSON list (closing bracket)
                end_idx = json_part.rfind("]")
                if end_idx != -1:
                    json_str = json_part[: end_idx + 1]
                    detections = json.loads(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse detections on step {i + 1}: {e}")

        if detections:
            labels = [d.get("label", "") for d in detections]
            logger.info(f"Visible objects: {labels}")

            # CALL SEMANTIC JUDGE
            match_result = semantic_match_target(object_name, labels)

            if match_result.get("found"):
                matched_label = match_result.get("match")
                logger.info(
                    f"SUCCESS: SEMANTIC MATCH: '{object_name}' matched with '{matched_label}'"
                )
                return f"Found {object_name} (matched: {matched_label}) on attempt {i + 1}.\n{detection_result}"
            else:
                logger.info(
                    f"Target '{object_name}' not semantically found in {labels}"
                )
        else:
            logger.info("No objects detected in this view.")

        # 3. If not found, ask VLM for next direction
        if i < max_attempts - 1:
            logger.info(f"Object not found. Asking agent for next direction...")

            # Get the image we just captured (detect_objects saves it to cache)
            img_b64 = get_latest_image(session_id)
            if img_b64.startswith("Error"):
                logger.warning(
                    "Could not retrieve image for direction planning. Defaulting to 'right'."
                )
                next_direction = "right"
            else:
                prompt = f"""
                You are a Robot Navigation Agent looking for a '{object_name}'.
                It was NOT found in the current view.
                Based on the scene, where should the camera move next to find it?
                Options: 'left', 'right', 'up', 'down', 'turn around'.
                Return ONLY the direction word.
                """
                next_direction = _call_vision_model(
                    prompt, f"Where should I look for the {object_name}?", img_b64
                )

                # Clean up response
                next_direction = (
                    next_direction.strip()
                    .lower()
                    .replace("'", "")
                    .replace('"', "")
                    .replace(".", "")
                )
                # Fallback if model is chatty
                valid_directions = ["left", "right", "up", "down", "turn around"]
                if not any(d in next_direction for d in valid_directions):
                    logger.warning(
                        f"Model returned unclear direction: '{next_direction}'. Defaulting to 'right'."
                    )
                    next_direction = "right"

            logger.info(f"Agent decided to move: {next_direction}")
            move_camera(next_direction)

    return f"Could not find {object_name} after {max_attempts} attempts."


# TODO: Add code execution tool

# Load tools from the tools package
load_tools(mcp)

if __name__ == "__main__":
    # Run the server
    mcp.run()
