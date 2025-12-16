import logging
import base64
import numpy as np
from PIL import Image
import io
import cv2
import os

logger = logging.getLogger("robot_mcp")

def get_image_resized(img_path):
    """
    Resize image to 1000px width while maintaining aspect ratio.
    """
    img = Image.open(img_path)
    img = img.resize(
        (1000, int(1000 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS
    )
    return img

def save_image_for_analysis(img):
    try:
        directory = "./camera"
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "received_image.jpeg")
        img.save(path, format="JPEG")
        logger.info(f"Saved image to {path} for analysis")
    except Exception as e:
        logger.error(f"Failed to save image for analysis: {e}")

def register(mcp):
    # @mcp.tool()
    # def get_camera_image() -> str:
    #     """
    #     Obtain an image from the robot's camera.
    #     Returns a base64 encoded JPEG string of the image.
    #     """
    #     logger.info("EXECUTING: get_camera_image")
        
    #     # Generate a dummy image (random noise or solid color)
    #     # In a real scenario, this would capture from cv2.VideoCapture
    #     img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #     img = Image.fromarray(img_array)
        
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="JPEG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    #     return img_str

    @mcp.tool()
    def resize_and_optimize_image(image_path: str) -> str:
        """
        Resize an image to 800px width for faster rendering and smaller API calls.
        Returns the path to the resized image.
        """
        try:
            img = get_image_resized(image_path)
            
            # Generate new path
            directory, filename = os.path.split(image_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_resized{ext}"
            new_path = os.path.join(directory, new_filename)
            
            img.save(new_path)
            return f"Image resized and saved to: {new_path}"
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return f"Error: {e}"
            
    # DEPRECATED: Use get_camera_image(camera_type="head") instead
    # @mcp.tool()
    # def get_head_camera_image() -> str:
    #     """
    #     Obtain an image from the robot's head camera (ZED 2).
    #     Returns a base64 encoded JPEG string of the image.
    #     """
    #     logger.info("EXECUTING: get_head_camera_image")
        
    # # Initialize camera (0 is usually default webcam)
    # cap = cv2.VideoCapture(6)  # change to correct camera

    # if not cap.isOpened():
    #     logger.error("Could not open camera")
    #     return "Error: Could not open camera."

    # try:
    #     # Allow camera to warm up
    #     for _ in range(15):  # Warm up more frames
    #         cap.read()

    #     ret, frame = cap.read()

    #     if not ret:
    #         logger.error("Failed to capture frame")
    #         return "Error: Failed to capture frame."

    #     # TODO: Refactor code; perhaps creating ABC for camera types?
    #     # Handle ZED Camera (Side-by-Side Stereo)
    #     # If the aspect ratio is 2:1 or wider, it's likely a stereo image.
    #     height, width, _ = frame.shape
    #     if width > height * 1.8:  # Simple heuristic for side-by-side
    #         # Crop to get just the left eye (first half of width)
    #         frame = frame[:, : width // 2, :]
    #         logger.info("Detected stereo image: Cropped to left eye.")

    #     # Convert BGR (OpenCV) to RGB (PIL)
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(rgb_frame)

    #     # Optimization: Resize to max 800px width if larger
    #     if img.size[0] > 800:
    #         new_height = int(800 * img.size[1] / img.size[0])
    #         img = img.resize((800, new_height), Image.Resampling.LANCZOS)
    #         logger.info(f"Image resized to 800x{new_height} for optimization")

    #     # Save to outputs folder with timestamp
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     filename = get_output_path(f"captured_image_{timestamp}.jpg", session_id)
    #     img.save(filename)
    #     logger.info(f"Image saved to {filename}")

    #     # Convert to Base64
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="JPEG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    #     return img_str

    # finally:
    #     cap.release()

    # DEPRECATED: Use get_camera_image(camera_type="right_wrist") instead
    # @mcp.tool()
    # def get_right_wrist_camera_image() -> str:
    #     """
    #     Obtain an image from the robot's right arms wrist camera (RealSense).
    #     Returns a base64 encoded JPEG string of the image.
    #     """
    #     # logger.info("EXECUTING: get_right_wrist_camera_image")
        
    #     # # --- REAL IMPLEMENTATION (Using /dev/video*) ---
    #     # # # Find the correct index for Right Wrist (e.g., 2, 4, etc.)
    #     # cap = cv2.VideoCapture(6) 
    #     # if not cap.isOpened():
    #     #     logger.error("Could not open right wrist camera")
    #     #     return ""
        
    #     # ret, frame = cap.read()
    #     # cap.release()
        
    #     # if not ret:
    #     #     return ""
        
    #     # # Convert BGR (OpenCV) to RGB (PIL)
    #     # img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # --- DUMMY IMPLEMENTATION (FOR TESTING) ---
    #     # img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    #     # img = Image.fromarray(img_array)
    #     # save_image_for_analysis(img)
    #     # buffered = io.BytesIO()
    #     # img.save(buffered, format="JPEG")
    #     # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    #     # return img_str

    # DEPRECATED: Use get_camera_image(camera_type="left_wrist") instead
    # @mcp.tool()
    # def get_left_wrist_camera_image() -> str:
    #     """
    #     Obtain an image from the robot's left wrist camera (RealSense).
    #     Returns a base64 encoded JPEG string of the image.
    #     """
    #     logger.info("EXECUTING: get_left_wrist_camera_image")
        
    #     # --- REAL IMPLEMENTATION (Using /dev/video*) ---
    #     # # Find the correct index for Left Wrist (e.g., 4, 6, etc.)
    #     # cap = cv2.VideoCapture(4) 
    #     # if not cap.isOpened():
    #     #     logger.error("Could not open left wrist camera")
    #     #     return ""
    #     # 
    #     # ret, frame = cap.read()
    #     # cap.release()
    #     # 
    #     # if not ret:
    #     #     return ""
    #     # 
    #     # # Convert BGR (OpenCV) to RGB (PIL)
    #     # img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # --- DUMMY IMPLEMENTATION (FOR TESTING) ---
    #     # img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    #     # img = Image.fromarray(img_array)
    #     # buffered = io.BytesIO()
    #     # img.save(buffered, format="JPEG")
    #     # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    #     # return img_str
