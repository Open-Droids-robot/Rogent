import logging
import base64
import numpy as np
from PIL import Image
import io
import cv2
import os

logger = logging.getLogger("robot_mcp")


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
    def get_head_camera_image() -> str:
        """
        Obtain an image from the robot's head camera (ZED 2).
        Returns a base64 encoded JPEG string of the image.
        """
        logger.info("EXECUTING: get_head_camera_image")
        
        # --- REAL IMPLEMENTATION (Using /dev/video*) ---
        # # Note: ZED 2 usually outputs side-by-side stereo. We need to split it.
        # # You need to find the correct index (e.g., /dev/video0 -> 0)
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            logger.error("Could not open head camera")
            return ""
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return ""
        
        # ZED outputs [Left | Right] side-by-side
        height, width, _ = frame.shape
        # Crop to get just the left eye (first half of width)
        left_eye_frame = frame[:, :width//2, :]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        img_array = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2RGB)

        
        # --- DUMMY IMPLEMENTATION (FOR TESTING) ---
        # img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        save_image_for_analysis(img)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str

    @mcp.tool()
    def get_right_wrist_camera_image() -> str:
        """
        Obtain an image from the robot's right arms wrist camera (RealSense).
        Returns a base64 encoded JPEG string of the image.
        """
        logger.info("EXECUTING: get_right_wrist_camera_image")
        
        # --- REAL IMPLEMENTATION (Using /dev/video*) ---
        # # Find the correct index for Right Wrist (e.g., 2, 4, etc.)
        cap = cv2.VideoCapture(6) 
        if not cap.isOpened():
            logger.error("Could not open right wrist camera")
            return ""
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return ""
        
        # Convert BGR (OpenCV) to RGB (PIL)
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DUMMY IMPLEMENTATION (FOR TESTING) ---
        # img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        save_image_for_analysis(img)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str

    @mcp.tool()
    def get_left_wrist_camera_image() -> str:
        """
        Obtain an image from the robot's left wrist camera (RealSense).
        Returns a base64 encoded JPEG string of the image.
        """
        logger.info("EXECUTING: get_left_wrist_camera_image")
        
        # --- REAL IMPLEMENTATION (Using /dev/video*) ---
        # # Find the correct index for Left Wrist (e.g., 4, 6, etc.)
        # cap = cv2.VideoCapture(4) 
        # if not cap.isOpened():
        #     logger.error("Could not open left wrist camera")
        #     return ""
        # 
        # ret, frame = cap.read()
        # cap.release()
        # 
        # if not ret:
        #     return ""
        # 
        # # Convert BGR (OpenCV) to RGB (PIL)
        # img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DUMMY IMPLEMENTATION (FOR TESTING) ---
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
