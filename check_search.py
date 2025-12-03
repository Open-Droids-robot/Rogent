import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

print("Testing search_web fallback logic...")

# Ensure we are in the right directory for imports
sys.path.append(os.getcwd())

from src.robot_tools_server import search_web

try:
    # We expect ddgs to fail on Jetson, so this should trigger the fallback
    result = search_web("Open Droids Jetson")
    print("\n--- Result ---")
    print(result)
    print("--------------\n")
    
    if "Google" in result or "Search Results" in result:
        print("SUCCESS: Search returned results.")
    elif "No results" in result:
        print("SUCCESS: Search ran but found nothing (this is okay, logic worked).")
    else:
        print("WARNING: Unexpected result format.")
        
except Exception as e:
    print(f"ERROR: search_web failed: {e}")
