import asyncio
import os
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_test():
    # Path to your server script
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    # Configure the server (running with python from .venv)
    python_exe = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python" # Fallback
        
    server_params = StdioServerParameters(
        command=python_exe,
        args=[server_script],
        env=None
    )

    print("🚀 Starting local MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # 1. List available tools
            tools = await session.list_tools()
            print("\n🛠️ Available Tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # 2. Test 'list_videos' tool
            print("\n📂 Testing 'list_videos'...")
            video_list = await session.call_tool("list_videos")
            print(f"Server Response:\n{video_list.content[0].text}")

            # 3. Test 'search_video_events' tool
            query = "person"
            print(f"\n🔍 Testing 'search_video_events' with query: '{query}'...")
            search_results = await session.call_tool("search_video_events", arguments={"query": query})
            print(f"Server Response:\n{search_results.content[0].text}")

            # 4. Test 'get_video_summary' tool
            print("\n📊 Testing 'get_video_summary'...")
            summary = await session.call_tool("get_video_summary", arguments={"video_name": "test.mp4"})
            print(f"Server Response:\n{summary.content[0].text}")

            # 5. Test 'generate_evidence_clip' tool (simulated)
            print("\n🎬 Testing 'generate_evidence_clip'...")
            clip = await session.call_tool("generate_evidence_clip", arguments={"video_name": "test.mp4", "timestamp_str": "00:10"})
            print(f"Server Response:\n{clip.content[0].text}")

            # 6. Test Resource Access
            print("\n📖 Testing resource access...")
            resources = await session.list_resources()
            if resources.resources:
                res = resources.resources[0]
                print(f"Reading resource: {res.uri}")
                content = await session.read_resource(res.uri)
                print(f"Content Length: {len(content.contents[0].text)} characters")
            else:
                print("No resources found (Try ingesting a video first).")

if __name__ == "__main__":
    try:
        asyncio.run(run_mcp_test())
    except Exception as e:
        print(f"❌ Error during MCP test: {e}")
