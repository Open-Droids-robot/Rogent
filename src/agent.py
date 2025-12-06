import asyncio
import logging
import sys
import os
import numpy as np
import base64
import io
from PIL import Image
from dotenv import load_dotenv
from colorama import Fore, Style, init


from typing import Annotated, TypedDict, List, Union, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from google.generativeai import protos

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from audio_manager import AudioManager
from transcriber import Transcriber
from synthesizer import Synthesizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Agent")

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in .env")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

import operator
import select


# ... (imports)

# --- LangGraph State ---
class AgentState(TypedDict):
    contents: Annotated[List[Any], operator.add] # Accumulate history

# --- Agent Graph ---
from contextlib import AsyncExitStack

# ... (previous imports)

# --- Agent Graph ---
class AgentGraph:
    def __init__(self, tool_map: dict, tools_schema: list):
        self.tool_map = tool_map
        self.tools_schema = tools_schema
        
        # Initialize Model
        model_name = os.getenv('GEMINI_MODEL')
        if not model_name:
             model_name = 'gemini-2.0-flash-exp'
             
        logger.info(f"Using Gemini Model: {model_name}")
        
        # Construct Tool with ONLY functions
        tool_config = protos.Tool(
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
            
            system_instruction = "\n\n".join(system_parts)
        except Exception as e:
            logger.error(f"Error loading system instructions: {e}")
            system_instruction = "You are a helpful robot assistant named Orin."

        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=[tool_config],
            system_instruction=system_instruction
        )

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
            # Pass the full history (contents) to generate_content_async
            response = await self.model.generate_content_async(contents)
            new_content = response.candidates[0].content
            return {"contents": [new_content]}
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return {"contents": [{"role": "model", "parts": [{"text": "I encountered an error."}]}]}

    async def tool_node(self, state: AgentState):
        contents = state['contents']
        last_content = contents[-1]
        
        parts = []
        if isinstance(last_content, dict):
            parts = last_content.get('parts', [])
        else:
            parts = last_content.parts

        new_parts = []
        for part in parts:
            # Handle both dict and object parts
            fn_call = None
            if isinstance(part, dict):
                if 'function_call' in part:
                    fn_call = part['function_call']
            else:
                if part.function_call:
                    fn_call = part.function_call
            
            if fn_call:
                # fn_call could be dict or object
                if isinstance(fn_call, dict):
                    tool_name = fn_call.get('name')
                    args = fn_call.get('args', {})
                else:
                    tool_name = fn_call.name
                    args = {k: v for k, v in fn_call.args.items()}
                
                # Helper to convert protobuf types to native python types
                def to_native(obj):
                    # Check for MapComposite (behaves like dict)
                    if hasattr(obj, 'items'):
                        return {k: to_native(v) for k, v in obj.items()}
                    # Check for RepeatedComposite (behaves like list)
                    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                        return [to_native(i) for i in obj]
                    else:
                        return obj

                # Convert args to native types
                args = to_native(args)
                
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
                        protos.Part(
                            function_response=protos.FunctionResponse(
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
                        result_text = result.content[0].text
                        
                    response_dict = {'result': result_text}
                    
                    new_parts.append(
                        protos.Part(
                            function_response=protos.FunctionResponse(
                                name=tool_name,
                                response=response_dict
                            )
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool Error: {e}")
                    new_parts.append(
                        protos.Part(
                            function_response=protos.FunctionResponse(
                                name=tool_name,
                                response={'error': str(e)}
                            )
                        )
                    )
        
        return {"contents": [{"role": "function", "parts": new_parts}]}

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
                if part.function_call:
                    return "continue"
        return "end"

    async def process(self, text: str, history: List[Any], image_data: str = None) -> (str, List[Any]):
        """
        Process user input with history and return (response_text, updated_history).
        """
        user_parts = [text]
        if image_data:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            user_parts.append(image)
            logger.info("Attached image to request.")
            
        # Create new user content
        new_user_content = {"role": "user", "parts": user_parts}
        
        current_contents = history + [new_user_content]
        inputs = {"contents": current_contents}
        
        final_state = await self.graph.ainvoke(inputs)
        
        # The final state 'contents' contains the full history including new interactions
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
                if part.text:
                    final_text += part.text
                
        return final_text, updated_history


# --- Main Agent ---
# --- Main Agent ---
class Agent:
    def __init__(self, robot_name="Orin"):
        self.robot_name = robot_name
        
        self.audio = AudioManager()
        self.transcriber = Transcriber(model_size="tiny.en", device="cpu", compute_type="int8")
        self.synthesizer = Synthesizer()
        self.graph = None 
        
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.is_listening = False
        
        self.history = [] # Maintain conversation history
        
    async def run(self):
        logger.info(f"Starting Agent {self.robot_name} (LangGraph + Multi-Server MCP)...")
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
                        
                        # Convert MCP tool to Gemini FunctionDeclaration
                        props = {}
                        required = []
                        if tool.inputSchema and 'properties' in tool.inputSchema:
                            for prop_name, prop_def in tool.inputSchema['properties'].items():
                                # Map JSON schema types to Gemini types
                                type_map = {
                                    "string": "STRING",
                                    "integer": "NUMBER", # Gemini uses NUMBER for both
                                    "number": "NUMBER",
                                    "boolean": "BOOLEAN",
                                    "array": "ARRAY",
                                    "object": "OBJECT"
                                }
                                p_type = type_map.get(prop_def.get('type'), "STRING")
                                prop_schema = {
                                    "type": p_type,
                                    "description": prop_def.get('description', '')
                                }
                                if p_type == "ARRAY":
                                    # Handle array items
                                    items_def = prop_def.get('items', {})
                                    item_type = type_map.get(items_def.get('type'), "STRING")
                                    prop_schema["items"] = {"type": item_type}
                                    
                                props[prop_name] = prop_schema
                            
                            if 'required' in tool.inputSchema:
                                required = tool.inputSchema['required']

                        fd = protos.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters={
                                'type': 'OBJECT',
                                'properties': props,
                                'required': required
                            }
                        )
                        tools_schema.append(fd)
                        
                except Exception as e:
                    logger.error(f"Failed to connect to {name} server: {e}")

            if not tool_map:
                logger.error("No tools loaded! Exiting.")
                return

            # Initialize Graph with aggregated tools
            self.graph = AgentGraph(tool_map, tools_schema)
            
            logger.info("Agent is READY. Speak into the microphone or type text.")
            
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
            # Pass history to process
            response_text, updated_history = await self.graph.process(text, self.history)
            
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
        logger.info("Stopping Agent...")
        agent.audio.stop()
