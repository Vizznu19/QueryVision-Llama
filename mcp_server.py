import os
import sqlite3
import numpy as np
from mcp.server.fastmcp import FastMCP
from core import (
    load_models, load_faiss_index, get_db_connection, 
    search_logs, cut_clip
)

# Initialize FastMCP server
mcp = FastMCP("QueryVision")

import sys
# Global state for models and index
# We load these on startup to keep the tools fast
print(" [MCP] Loading models and index, please wait...", file=sys.stderr)
yolo, blip_processor, blip_model, embedder, device = load_models()
conn = get_db_connection()
vector_index = load_faiss_index(conn)
print(" [MCP] Server ready!", file=sys.stderr)

@mcp.tool()
def search_video_events(query: str, limit: int = 5) -> str:
    """
    Search through video logs using natural language to find specific objects, 
    people, or events. Returns matching timestamps and descriptions.
    """
    results = search_logs(query, embedder, vector_index, conn, limit=limit)
    
    if not results:
        return "No matching events found in the database."

    output = f"Found {len(results)} matching events:\n"
    for score, (video, ts, cap) in results:
        m, s = divmod(int(ts), 60)
        output += f"- [{m:02d}:{s:02d}] {video}: {cap} (Similarity: {score:.2f})\n"
    
    return output

@mcp.tool()
def list_videos() -> str:
    """
    List all videos that have been processed and indexed in the system.
    """
    c = conn.cursor()
    c.execute("SELECT file_name, processed_date FROM processed_videos")
    rows = c.fetchall()
    
    if not rows:
        return "No videos have been processed yet."
    
    output = "Indexed Videos:\n"
    for name, date in rows:
        output += f"- {name} (Processed on: {date})\n"
    return output

@mcp.resource("logs://{video_name}")
def get_video_logs(video_name: str) -> str:
    """
    Get the full chronological log of all events detected in a specific video.
    """
    c = conn.cursor()
    c.execute("SELECT timestamp, caption FROM logs WHERE video_name = ? ORDER BY timestamp ASC", (video_name,))
    rows = c.fetchall()
    
    if not rows:
        return f"No logs found for video: {video_name}"
    
    output = f"Chronological logs for {video_name}:\n"
    for ts, cap in rows:
        m, s = divmod(int(ts), 60)
        output += f"[{m:02d}:{s:02d}] {cap}\n"
    return output

@mcp.tool()
def get_video_summary(video_name: str) -> str:
    """
    Get a high-level summary of the events and objects detected in a specific video.
    """
    c = conn.cursor()
    c.execute("SELECT caption FROM logs WHERE video_name = ?", (video_name,))
    rows = c.fetchall()
    
    if not rows:
        return f"No data found for video: {video_name}"
    
    # Simple statistical summary
    unique_objects = set()
    for (cap,) in rows:
        # Extract the class name (it's formatted as "Class: Caption")
        if ":" in cap:
            obj = cap.split(":")[0].strip().lower()
            unique_objects.add(obj)
    
    output = f"Summary for {video_name}:\n"
    output += f"- Total events logged: {len(rows)}\n"
    output += f"- Unique objects detected: {', '.join(unique_objects)}\n"
    output += "- Use 'logs://{video_name}' to see the full timeline."
    return output

@mcp.tool()
def generate_evidence_clip(video_name: str, timestamp_str: str) -> str:
    """
    Generate a 20-second evidence video clip from a source video at a specific timestamp.
    timestamp_str format: "MM:SS" or total seconds as string.
    """
    try:
        if ":" in timestamp_str:
            m, s = map(int, timestamp_str.split(":"))
            total_seconds = m * 60 + s
        else:
            total_seconds = float(timestamp_str)
            
        clip_path = cut_clip(video_name, total_seconds)
        if clip_path:
            return f"✅ Evidence clip generated successfully: {os.path.basename(clip_path)}"
        else:
            return "❌ Error: Could not generate clip. Check if the video exists."
    except Exception as e:
        return f"❌ Error processing timestamp: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
