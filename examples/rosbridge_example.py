#!/usr/bin/env python3
"""
ROS Bridge Connection Example

This script demonstrates how to communicate with a ROS system via rosbridge_server using WebSockets.
It performs the following operations:
1. Connects to the ROS Bridge WebSocket server.
2. Calls a service to list available ROS topics.
3. Publishes a command to move a servo.
"""

import json
import time
import sys

# Check for websocket-client library
try:
    import websocket
except ImportError:
    print("Error: The 'websocket-client' library is required.")
    print("Please install it using: pip install websocket-client")
    sys.exit(1)

# Configuration
ROS_BRIDGE_URI = "ws://127.0.0.1:9090"

def run_example():
    """Main function to run the example interactions with ROS Bridge."""
    print(f"Attempting to connect to ROS Bridge at {ROS_BRIDGE_URI}...")
    
    try:
        # 1. Establish WebSocket Connection
        # We use a timeout to fail fast if the server isn't running
        ws = websocket.create_connection(ROS_BRIDGE_URI, timeout=3.0)
        print("✅ Connected to ROS Bridge!")
        
        # 2. Example: Call a Service (Listing Topics)
        # The 'call_service' operation allows us to invoke ROS services.
        # Here we call /rosapi/topics to get a list of all active topics.
        print("\n[1/2] Listing Topics via /rosapi/topics...")
        service_request = {
            "op": "call_service",
            "service": "/rosapi/topics",
            "type": "rosapi/Topics",
            "id": "list_topics_req"
        }
        ws.send(json.dumps(service_request))
        
        # Receive the service response
        result = ws.recv()
        # Just printing the first 100 chars to verify receipt without cluttering output
        print(f"Response received (preview): {result[:100]}...")

        # 3. Example: Publish a Message (Move Servo)
        # To publish, we follow the pattern: Advertise -> Publish -> Unadvertise
        print("\n[2/2] Moving Head (Tilt=500)...")
        topic = "/servo_control/move"
        msg_type = "servo_interfaces/msg/ServoMove"
        
        # A. Advertise: Tell rosbridge we intend to publish on this topic
        ws.send(json.dumps({
            "op": "advertise", 
            "topic": topic, 
            "type": msg_type
        }))
        
        # B. Publish: Send the actual message data
        # 'servo_id': 1 is typically the head tilt servo
        # 'angle': 500 is the center position
        ws.send(json.dumps({
            "op": "publish", 
            "topic": topic, 
            "msg": {"servo_id": 1, "angle": 500}
        }))
        print(f"Message published to {topic}!")
        
        # Wait a moment to ensure message is processed
        time.sleep(1)
        
        # C. Unadvertise: Clean up our registration
        ws.send(json.dumps({
            "op": "unadvertise", 
            "topic": topic
        }))
        
        # 4. Close the WebSocket connection
        ws.close()
        print("\n✅ Success! The example finished without errors.")
        
    except ConnectionRefusedError:
        print(f"\n❌ Connection Refused! The server at {ROS_BRIDGE_URI} is not running.")
        print("   - Check rosbridge.log for crashes")
        print("   - Ensure you have launched rosbridge_server")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_example()
