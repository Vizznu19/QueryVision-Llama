import os
import sqlite3
import hashlib
import numpy as np
import faiss
import torch
import cv2
import subprocess
import uuid
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
DB_FILE = os.path.join(BASE_DIR, "queryvision.db")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_store.index")
VIDEO_FOLDER = os.path.join(BASE_DIR, "project_data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "search_output")

# Ensure directories exist
for folder in [VIDEO_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, video_hash TEXT, video_name TEXT, timestamp REAL, caption TEXT, embedding BLOB)''') 
    c.execute('''CREATE TABLE IF NOT EXISTS processed_videos (file_hash TEXT PRIMARY KEY, file_name TEXT, processed_date TEXT)''')
    conn.commit()
    return conn

def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# --- LOAD MODELS ---
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use your custom model if you have trained one, otherwise 'yolov8n.pt'
    yolo = YOLO('yolov8m.pt') 
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=CACHE_DIR).to(device)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR, device='cpu')
    return yolo, processor, model, embedder, device

# --- HELPER FUNCTIONS ---
def rebuild_faiss_index(conn):
    c = conn.cursor()
    c.execute("SELECT embedding FROM logs ORDER BY id ASC")
    rows = c.fetchall()
    new_index = faiss.IndexFlatIP(384)
    if rows:
        embeddings = [np.frombuffer(row[0], dtype='float32') for row in rows]
        new_index.add(np.stack(embeddings))
    faiss.write_index(new_index, FAISS_INDEX_FILE)
    return new_index

def load_faiss_index(conn):
    if os.path.exists(FAISS_INDEX_FILE):
        return faiss.read_index(FAISS_INDEX_FILE)
    else:
        return rebuild_faiss_index(conn)

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def search_logs(query, embedder, vector_index, conn, limit=10):
    query_vector = embedder.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = vector_index.search(query_vector, 20)
    
    scored_logs = []
    c = conn.cursor()
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        c.execute("SELECT video_name, timestamp, caption FROM logs ORDER BY id ASC LIMIT 1 OFFSET ?", (int(idx),))
        row = c.fetchone()
        if row:
            scored_logs.append((float(score), row))
            
    scored_logs.sort(key=lambda x: x[0], reverse=True)
    return scored_logs[:limit]

def cut_clip(video_name, timestamp, duration=20):
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    start_time = max(0, timestamp - 10)
    unique_id = str(uuid.uuid4())[:8]
    clip_filename = f"evidence_{int(timestamp)}s_{unique_id}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, clip_filename)
    
    ffmpeg_cmd = "ffmpeg"
    local_ffmpeg = os.path.join(BASE_DIR, "ffmpeg.exe")
    if os.path.exists(local_ffmpeg): ffmpeg_cmd = local_ffmpeg
    
    if not os.path.exists(video_path): return None

    # H.264 Re-encoding for Browser Compatibility
    command = [ffmpeg_cmd, "-ss", str(start_time), "-i", video_path, "-t", str(duration), 
               "-c:v", "libx264", "-c:a", "aac", "-y", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
