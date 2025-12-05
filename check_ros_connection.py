import json
import websocket
import time
import threading

def on_message(ws, message):
    data = json.loads(message)
    if "values" in data and "topics" in data["values"]:
        print(f"SUCCESS: Received {len(data['values']['topics'])} topics")
    else:
        print(f"Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Closed")

def on_open(ws):
    print("Connected")
    # Call rosapi/topics
    msg = {
        "op": "call_service",
        "service": "/rosapi/topics",
        "type": "rosapi/Topics",
        "id": "test_topics"
    }
    ws.send(json.dumps(msg))

if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://127.0.0.1:9090",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    
    # Run in a thread so we can kill it after a few seconds
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    time.sleep(5)
    ws.close()
