import asyncio
import argparse
import json
import sys
from robot_tools_server import mcp

async def main():
    parser = argparse.ArgumentParser(description="Run an MCP tool by name.")
    parser.add_argument("tool_name", help="Name of the tool to run")
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    
    args = parser.parse_args()
    
    tool_args = {}
    for arg in args.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to parse value as JSON (for booleans, numbers, etc.)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass # Keep as string
            tool_args[key] = value
        else:
            print(f"Warning: Ignoring argument '{arg}' (not in key=value format)")

    print(f"Running tool: {args.tool_name}")
    print(f"Arguments: {tool_args}")

    try:
        # Call the tool
        result = await mcp.call_tool(args.tool_name, arguments=tool_args)
        
        print("\n--- Result ---")
        # FastMCP call_tool returns a tuple (content_list, context) or just content list
        content_list = result
        if isinstance(result, tuple):
            content_list = result[0]
            
        if isinstance(content_list, list):
             for item in content_list:
                 # Check for TextContent
                 if hasattr(item, 'text'):
                     text_content = item.text
                     # Check if it looks like a base64 image
                     # Simple heuristic: long string, no spaces, valid base64
                     if len(text_content) > 100 and " " not in text_content[:100]:
                         try:
                             import base64
                             import binascii
                             
                             # Try to decode
                             image_data = base64.b64decode(text_content, validate=True)
                             
                             # Save to file
                             output_filename = "output_image.jpg"
                             with open(output_filename, "wb") as f:
                                 f.write(image_data)
                             print(f"\n[SUCCESS] Image saved to {output_filename}")
                             print(f"(Base64 length: {len(text_content)})")
                             
                         except (binascii.Error, ValueError):
                             # Not a valid base64 string or just text
                             print(f"{text_content}")
                     else:
                         print(f"{text_content}")

                 # Check for ImageContent or other types if needed
                 elif hasattr(item, 'data'):
                     print(f"[Binary Data: {len(item.data)} bytes]")
                 else:
                     print(f"{item}")
        else:
            print(result)
        
    except Exception as e:
        print(f"\nError running tool: {e}")

if __name__ == "__main__":
    asyncio.run(main())
