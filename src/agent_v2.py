import asyncio
import logging
import sys
import os
import numpy as np
import base64
import io
import uuid
from PIL import Image
from dotenv import load_dotenv
from colorama import Fore, Style, init


from typing import Annotated, TypedDict, List, Union, Any
from langgraph.graph import StateGraph, END

# Modern Google GenAI SDK
from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from audio_manager import AudioManager
from transcriber import Transcriber
from synthesizer import Synthesizer

import weave 

weave.init("test-agent-v2")

@weave.op()
def trace_image(image: Image.Image, label: str = "tool_image"):
    """Helper to capture images in Weave traces."""
    return "Image Logged"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentV2")

# Configure Gemini with modern SDK
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in .env")

import operator
import select


# --- LangGraph State ---
class AgentState(TypedDict):
    contents: Annotated[List[Any], operator.add] # Accumulate history (List[types.Content])

# --- Helper: Schema Conversion ---
def mcp_schema_to_gemini_schema(schema: dict) -> types.Schema:
    """Recursively convert a dict schema (MCP/JSON Schema) to types.Schema."""
    if not schema:
        return None
    
    # Extract type and ensure uppercase for Gemini Enum
    type_val = schema.get('type')
    if isinstance(type_val, str):
        type_val = type_val.upper()
    
    # Handle properties recursively
    properties = {}
    if 'properties' in schema and isinstance(schema['properties'], dict):
        for k, v in schema['properties'].items():
            properties[k] = mcp_schema_to_gemini_schema(v)
            
    # Handle items recursively (for arrays)
    items = None
    if 'items' in schema and isinstance(schema['items'], dict):
        items = mcp_schema_to_gemini_schema(schema['items'])
        
    return types.Schema(
        type=type_val,
        description=schema.get('description'),
        properties=properties if properties else None,
        required=schema.get('required'),
        items=items
    )

# --- Agent Graph (Modernized) ---
from contextlib import AsyncExitStack

class AgentGraph:
    def __init__(self, tool_map: dict, tools_schema: list):
        self.tool_map = tool_map
        self.tools_schema = tools_schema
        
        # Initialize the modern GenAI client
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Get model name
        model_name = os.getenv('GEMINI_MODEL')
        if not model_name:
             model_name = 'gemini-2.0-flash-exp'
             
        self.model_name = model_name
        logger.info(f"Using Gemini Model: {model_name}")
        
        # Construct Tool with modern types
        self.tool_config = types.Tool(
            function_declarations=self.tools_schema
        )
        
        # Load System Instruction from files
        try:
            # Determine paths relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(base_dir)
            prompts_dir = os.path.join(project_root, "prompts")
            
            persona_path = os.path.join(prompts_dir, "persona.txt")
            perception_path = os.path.join(prompts_dir, "perception.txt")
            
            system_parts = []
            
            # Load persona.txt
            if os.path.exists(persona_path):
                with open(persona_path, "r") as f:
                    system_parts.append(f.read())
            else:
                logger.warning("prompts/persona.txt not found, using default.")
                system_parts.append("You are a helpful robot assistant named Orin.")
            
            # Load perception.txt
            if os.path.exists(perception_path):
                with open(perception_path, "r") as f:
                    system_parts.append(f.read())
            else:
                logger.warning("prompts/perception.txt not found.")
            
            self.system_instruction = "\n\n".join(system_parts)
        except Exception as e:
            logger.error(f"Error loading system instructions: {e}")
            self.system_instruction = "You are a helpful robot assistant named Orin."

        # Build Graph
        builder = StateGraph(AgentState)
        
        builder.add_node("agent", self.agent_node)
        builder.add_node("tools", self.tool_node)
        
        builder.set_entry_point("agent")
        
        builder.add_conditional_edges(
            "agent",
            self.should_continue,
            {"continue": "tools", "end": END}
        )
        
        builder.add_edge("tools", "agent")
        
        self.graph = builder.compile()

    async def agent_node(self, state: AgentState):
        contents = state['contents']
        try:
            # Use the modern SDK's async API
            # contents is a list of types.Content objects (or compatible dicts)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=[self.tool_config]
                )
            )
            # Response handling
            if not response.candidates:
                 logger.error("No candidates returned from Gemini.")
                 return {"contents": [types.Content(role="model", parts=[types.Part(text="I didn't get a response.")])]}

            new_content = response.candidates[0].content
            return {"contents": [new_content]}
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return {"contents": [types.Content(role="model", parts=[types.Part(text=f"I encountered an error: {e}")])]}

    @weave.op()
    async def tool_node(self, state: AgentState):
        contents = state['contents']
        last_content = contents[-1]
        
        # Extract parts from the last content (safely handling dict vs object)
        parts = []
        if isinstance(last_content, dict):
            parts = last_content.get('parts', [])
        else:
            parts = last_content.parts

        new_parts = []
        for part in parts:
            # Handle both dict and object parts for function_call
            fn_call = None
            if isinstance(part, dict):
                if 'function_call' in part:
                    fn_call = part['function_call']
            else:
                if hasattr(part, 'function_call') and part.function_call:
                    fn_call = part.function_call
            
            if fn_call:
                # Extract name and args
                if isinstance(fn_call, dict):
                    tool_name = fn_call.get('name')
                    args = fn_call.get('args', {})
                else:
                    tool_name = fn_call.name
                    args = fn_call.args # In modern SDK, this is likely already a dict
                
                # Ensure args is a dict
                if not isinstance(args, dict):
                     # If it's some other object, try to convert (simplistic conversion)
                     try:
                         args = dict(args)
                     except:
                         pass

                # Handle Hallucinations / Mappings
                if tool_name == 'search':
                    logger.info("Mapping 'search' tool call to 'search_web'")
                    tool_name = 'search_web'
                    if 'queries' in args:
                        q = args.pop('queries')
                        if isinstance(q, list):
                            args['query'] = " ".join(q)
                        else:
                            args['query'] = str(q)
                
                logger.info(f"Executing Tool: {tool_name} with {args}")
                
                session = self.tool_map.get(tool_name)
                if not session:
                    logger.error(f"Tool {tool_name} not found in tool_map.")
                    new_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={'error': f"Tool {tool_name} not found."}
                            )
                        )
                    )
                    continue

                try:
                    result = await session.call_tool(tool_name, arguments=args)
                    result_text = ""
                    if hasattr(result, 'content') and result.content:
                        # Assuming result.content is list of TextContent or similar
                        # MCP content structure
                        if isinstance(result.content, list):
                             result_text = " ".join([c.text for c in result.content if hasattr(c, 'text')])
                        else:
                             result_text = str(result.content)
                    
                    # --- Trace Image for get_camera_image ---
                    if tool_name == "get_camera_image" and result_text and not result_text.startswith("Error"):
                        try:
                            # result_text is expected to be a base64 string
                            img_bytes = base64.b64decode(result_text)
                            img = Image.open(io.BytesIO(img_bytes))
                            trace_image(img, "camera_capture")
                        except Exception as e:
                            logger.warning(f"Failed to trace image: {e}")
                    # --------------------------------

                    # --- Trace Image for Plotting Tools ---
                    if tool_name in ["plot_detections", "plot_trajectory"] and "saved to" in result_text:
                        try:
                            # Extract filename from "Visualization saved to outputs/filename.jpg"
                            # We look for the last word or parse the path
                            parts = result_text.split()
                            filename = parts[-1] # Assuming filename is the last word
                            
                            # Clean up filename if it has punctuation
                            filename = filename.rstrip('.')
                            
                            if os.path.exists(filename):
                                img = Image.open(filename)
                                trace_image(img, label=f"{tool_name}_result")
                                logger.info(f"Logged {tool_name} image to Weave.")
                            else:
                                logger.warning(f"Could not find file {filename} to log to Weave.")
                                
                        except Exception as e:
                            logger.warning(f"Failed to trace plot image: {e}")
                    # -------------------------------------

                    if tool_name == "get_camera_image":
                         img_bytes = base64.b64decode(result_text)
                         blob = types.Blob(mime_type='image/jpeg', data=img_bytes)
                         
                         # Add function response AND the image blob
                         new_parts.append(
                             types.Part(
                                 function_response=types.FunctionResponse(
                                     name=tool_name,
                                     response={'result': 'Image Captured.'}
                                 )
                             )
                         )
                         new_parts.append(types.Part(inline_data=blob)) 
                    else:
                         # Standard text response
                         new_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={'result': result_text}
                                )
                            )
                        )
                except Exception as e:
                    logger.error(f"Tool Error: {e}")
                    new_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={'error': str(e)}
                            )
                        )
                    )
        
        return {"contents": [types.Content(role="function", parts=new_parts)]}

    def should_continue(self, state: AgentState):
        contents = state['contents']
        last_content = contents[-1]
        
        parts = []
        if isinstance(last_content, dict):
            parts = last_content.get('parts', [])
        else:
            parts = last_content.parts
            
        for part in parts:
            if isinstance(part, dict):
                if 'function_call' in part:
                    return "continue"
            else:
                if hasattr(part, 'function_call') and part.function_call:
                    return "continue"
        return "end"

    @weave.op()
    async def process(self, text: str, history: List[Any], image_data: str = None, session_id: str = None) -> (str, List[Any]):
        """
        Process user input with history and return (response_text, updated_history).
        """
        parts = []
        parts.append(types.Part(text=text))
        
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                # Create a Blob for the image
                blob = types.Blob(
                    mime_type='image/jpeg', # Defaulting to jpeg, could infer
                    data=image_bytes
                )
                parts.append(types.Part(inline_data=blob))
                logger.info("Attached image to request.")
            except Exception as e:
                logger.error(f"Failed to process image data: {e}")
            
        # Create new user content as a types.Content object
        new_user_content = types.Content(role="user", parts=parts)
        
        current_contents = history + [new_user_content]
        inputs = {"contents": current_contents}
        
        final_state = await self.graph.ainvoke(inputs)
        
        # The final state 'contents' contains the full history
        updated_history = final_state['contents']
        
        # Extract text from the last message
        last_content = updated_history[-1]
        final_text = ""
        
        parts = []
        if isinstance(last_content, dict):
            parts = last_content.get('parts', [])
        else:
            parts = last_content.parts
            
        for part in parts:
            if isinstance(part, dict):
                if 'text' in part:
                    final_text += part['text']
            else:
                if hasattr(part, 'text') and part.text:
                    final_text += part.text
                
        return final_text, updated_history


# --- Main Agent ---
class Agent:
    def __init__(self, robot_name="Orin"):
        self.robot_name = robot_name
        self.session_id = str(uuid.uuid4()) # Generate unique ID for this entire run
        
        self.audio = AudioManager()
        self.transcriber = Transcriber(model_size="tiny.en", device="cpu", compute_type="int8")
        self.synthesizer = Synthesizer()
        self.graph = None 
        
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.is_listening = False
        
        self.history = [] # Maintain conversation history
        
    async def run(self):
        logger.info(f"Starting Agent V2 {self.robot_name} (LangGraph + Modern GenAI SDK + Multi-Server MCP)...")
        self.audio.start()
        
        # Determine paths relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        
        robot_server_path = os.path.join(base_dir, "robot_tools_server.py")
        ros_server_path = os.path.join(project_root, "ros-mcp-server", "server.py")
        
        # Define servers to connect to
        servers = {
            "local": {
                "command": sys.executable,
                "args": [robot_server_path],
                "env": {**os.environ.copy(), "PYTHONPATH": base_dir}
            },
            "ros": {
                "command": sys.executable,
                "args": [ros_server_path], 
                "env": {
                    **os.environ.copy(),
                    "PYTHONPATH": os.path.join(project_root, "ros-mcp-server")
                }
            }
        }

        async with AsyncExitStack() as stack:
            tool_map = {}
            tools_schema = []

            for name, config in servers.items():
                logger.info(f"Connecting to {name} MCP server...")
                # Verify file exists before connecting
                if not os.path.exists(config["args"][0]):
                    logger.warning(f"Server script not found at {config['args'][0]}. Skipping {name} server.")
                    continue

                try:
                    server_params = StdioServerParameters(
                        command=config["command"],
                        args=config["args"],
                        env=config["env"]
                    )
                    
                    # Create client (stdio_client returns a context manager)
                    read, write = await stack.enter_async_context(stdio_client(server_params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    
                    # List tools
                    result = await session.list_tools()
                    logger.info(f"Connected to {name} server. Found {len(result.tools)} tools.")
                    
                    for tool in result.tools:
                        logger.info(f"  - Adding tool: {tool.name} (from {name})")
                        tool_map[tool.name] = session
                        
                        # Convert MCP tool inputSchema to Gemini Schema
                        gemini_schema = None
                        if tool.inputSchema:
                             gemini_schema = mcp_schema_to_gemini_schema(tool.inputSchema)

                        # Use modern types.FunctionDeclaration
                        # parameters expects a types.Schema object
                        fd = types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=gemini_schema
                        )
                        tools_schema.append(fd)
                        
                except Exception as e:
                    logger.error(f"Failed to connect to {name} server: {e}")

            if not tool_map:
                logger.error("No tools loaded! Please check if server scripts exist and are running.")
                # We can continue but agent won't have tools
                
            # Initialize Graph with aggregated tools
            self.graph = AgentGraph(tool_map, tools_schema)
            
            logger.info("Agent V2 is READY. Speak into the microphone or type text.")
            
            while True:
                # Check stdin
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        if line:
                            text = line.strip()
                            if text:
                                logger.info(f"Text Input: '{text}'")
                                await self.process_input(text)
                except Exception:
                    pass

                while not self.audio.audio_queue.empty():
                    frame_bytes = self.audio.audio_queue.get()
                    is_speech = self.audio.vad.is_speech(frame_bytes, self.audio.sample_rate)
                    
                    if is_speech:
                        if not self.is_listening:
                            logger.info("Speech detected...")
                            self.is_listening = True
                        self.audio_buffer.extend(frame_bytes)
                        self.silence_frames = 0
                    else:
                        if self.is_listening:
                            self.silence_frames += 1
                            if self.silence_frames > 33:
                                logger.info("Silence detected. Processing speech...")
                                await self.process_speech()
                                self.reset_listening()
                
                await asyncio.sleep(0.01)

    def reset_listening(self):
        self.is_listening = False
        self.audio_buffer = bytearray()
        self.silence_frames = 0

    async def process_speech(self):
        audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        text = self.transcriber.transcribe(audio_data)
        if not text:
            return
            
        logger.info(f"Transcribed: '{text}'")
        await self.process_input(text)

    async def process_input(self, text):
        try:
            if not self.graph:
                 await self.speak("I am not fully initialized yet.")
                 return

            # Pass history and session_id to process
            response_text, updated_history = await self.graph.process(text, self.history, session_id=self.session_id)
            
            # Update history
            self.history = updated_history
            
            logger.info(f"Agent Response: {response_text}")
            
            if response_text:
                await self.speak(response_text)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Graph Execution Error: {e}")
            await self.speak("I encountered an error.")

    async def speak(self, text):
        self.audio.set_speaking_state(True)
        await self.synthesizer.speak(text)
        self.audio.set_speaking_state(False)

if __name__ == "__main__":
    agent = Agent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Stopping Agent V2...")
        agent.audio.stop()
