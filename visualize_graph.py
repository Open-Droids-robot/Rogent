import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent import AgentGraph
from mcp import ClientSession

# Mock session since we only need the graph structure
class MockSession:
    pass

def visualize():
    print("Building graph...")
    # Initialize with mock session
    agent_graph = AgentGraph(MockSession())
    
    print("Generating visualization...")
    try:
        # Get the graph drawable
        graph = agent_graph.graph
        
        # Generate PNG
        png_bytes = graph.get_graph().draw_mermaid_png()
        
        output_file = "agent_graph.png"
        with open(output_file, "wb") as f:
            f.write(png_bytes)
            
        print(f"Graph saved to {output_file}")
        
        # Also generate a Mermaid file with tools list
        print("Generating detailed Mermaid file...")
        mermaid_code = graph.get_graph().draw_mermaid()
        
        # Extract tool names
        tool_names = [t.name for t in agent_graph.tools_schema]
        tools_list = "\\n".join(tool_names)
        
        # Append a legend node
        mermaid_code += f"\n\n    subgraph Tools\n    direction TB\n    T[Available Tools:\\n{tools_list}]\n    end"
        
        with open("agent_graph.mmd", "w") as f:
            f.write(mermaid_code)
        print("Detailed graph saved to agent_graph.mmd")
        
        # Render PNG using mermaid.ink
        print("Rendering PNG via mermaid.ink...")
        import base64
        import requests
        
        graphbytes = mermaid_code.encode("utf8")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        
        url = "https://mermaid.ink/img/" + base64_string
        
        response = requests.get(url)
        if response.status_code == 200:
            with open("agent_graph_with_tools.png", "wb") as f:
                f.write(response.content)
            print("Graph with tools saved to agent_graph_with_tools.png")
        else:
            print(f"Failed to render PNG via mermaid.ink: {response.status_code}")
            
    except Exception as e:
        print(f"Error visualizing graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize()
