import streamlit as st
import os
import cv2
import sqlite3
import subprocess
import ollama
import torch
import hashlib
import pickle
import numpy as np
import faiss
import re
import face_recognition
import uuid
import shutil
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

# --- PAGE CONFIG ---
st.set_page_config(page_title="QueryVision: AI Forensic Video Analyst", page_icon="👁️", layout="wide")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
DB_FILE = os.path.join(BASE_DIR, "queryvision.db")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_store.index")
VIDEO_FOLDER = os.path.join(BASE_DIR, "project_data")
FACES_FOLDER = os.path.join(BASE_DIR, "project_faces") 
OUTPUT_FOLDER = os.path.join(BASE_DIR, "search_output")

# Ensure directories exist
for folder in [VIDEO_FOLDER, OUTPUT_FOLDER, FACES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, video_hash TEXT, video_name TEXT, timestamp REAL, caption TEXT, embedding BLOB)''') 
    c.execute('''CREATE TABLE IF NOT EXISTS processed_videos (file_hash TEXT PRIMARY KEY, file_name TEXT, processed_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, video_hash TEXT, video_name TEXT, timestamp REAL, image_path TEXT, encoding BLOB)''')
    conn.commit()
    return conn

conn = init_db()

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"🚀 AI Hardware: {device.upper()}")
    yolo = YOLO('yolov8n.pt') 
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR).to(device)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR, device='cpu')
    return yolo, processor, model, embedder, device

yolo, blip_processor, blip_model, embedder, device = load_models()

# --- HELPER FUNCTIONS ---
def rebuild_faiss_index():
    c = conn.cursor()
    c.execute("SELECT embedding FROM logs ORDER BY id ASC")
    rows = c.fetchall()
    new_index = faiss.IndexFlatL2(384)
    if rows:
        embeddings = [np.frombuffer(row[0], dtype='float32') for row in rows]
        new_index.add(np.stack(embeddings))
    faiss.write_index(new_index, FAISS_INDEX_FILE)
    return new_index

if os.path.exists(FAISS_INDEX_FILE):
    vector_index = faiss.read_index(FAISS_INDEX_FILE)
else:
    vector_index = rebuild_faiss_index()

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cut_clip(video_name, timestamp, duration=10):
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    start_time = max(0, timestamp - 5)
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

# --- UI LAYOUT ---
st.title("👁️ QueryVision : AI Forensic Video Analyst")

c = conn.cursor()
log_count = c.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
face_count = c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

st.sidebar.metric("Events Tracked", log_count)
st.sidebar.metric("Unique Faces Found", face_count)

tab_search, tab_faces, tab_upload, tab_manage = st.tabs(["🕵️ Search Analyst", "👥 Face Gallery", "📂 Smart Ingest", "🗑️ Manage"])

# --- TAB 1: SEARCH ---
with tab_search:
    query = st.text_area("Type your query here:", placeholder="e.g. Find the man in the blue shirt.")
    if st.button("Run Analysis", type="primary"):
        if log_count == 0:
            st.error("Database empty.")
        else:
            with st.spinner("Searching Vector Database..."):
                query_vector = embedder.encode([query])
                distances, indices = vector_index.search(np.array(query_vector).astype('float32'), 50)
                
                relevant_logs = []
                for idx in indices[0]:
                    if idx == -1: continue
                    c.execute(f"SELECT video_name, timestamp, caption FROM logs ORDER BY id ASC LIMIT 1 OFFSET {idx}")
                    row = c.fetchone()
                    if row: relevant_logs.append(row)

            context_log = ""
            for video, ts, cap in relevant_logs:
                m, s = divmod(int(ts), 60)
                context_log += f"- [{m:02d}:{s:02d}] {video}: {cap}\n"
            
            with st.spinner("Llama 3 Reasoning..."):
                try:
                    system_prompt = """
                    You are QueryVision, an expert Forensic Video Analyst. 
                    Your task is to analyze structured computer vision logs to find the exact moment a specific event occurred.
                    **ANALYSIS RULES:**
                    1. **Semantic Matching:** Look for visual descriptions that match the user's intent.
                    2. **Time Selection:** Select the **earliest** timestamp (the start of the event).
                    3. **Strict Formatting:** You must output the final determination in square brackets like `[MM:SS]`.
                    **OUTPUT FORMAT:**
                    Provide a single sentence reasoning, followed strictly by the timestamp. Example: "Subject identified matching description. [04:12]"
                    """
                    response = ollama.chat(model='llama3.2', messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"LOGS:\n{context_log}\n\nQUESTION: {query}"}
                    ])
                    answer = response['message']['content']
                    st.markdown(answer)
                    
                    time_matches = re.findall(r'(\d+):(\d+)', answer)
                    if time_matches:
                        minutes, seconds = map(int, time_matches[-1])
                        total_seconds = (minutes * 60) + seconds
                        best_video = relevant_logs[0][0]
                        st.divider()
                        st.subheader(f"🎥 Evidence Clip ({minutes:02d}:{seconds:02d})")
                        clip_path = cut_clip(best_video, total_seconds)
                        if clip_path and os.path.exists(clip_path):
                            st.video(clip_path)
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: FACE GALLERY ---
with tab_faces:
    st.header("Identified Subjects")
    c.execute("SELECT image_path, timestamp, video_name FROM faces ORDER BY id DESC")
    faces = c.fetchall()
    if not faces:
        st.info("No faces detected yet.")
    else:
        cols = st.columns(5)
        for i, (img_path, ts, vid_name) in enumerate(faces):
            with cols[i % 5]:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    m, s = divmod(int(ts), 60)
                    st.caption(f"Time: {m:02d}:{s:02d}\nFile: {vid_name}")

# --- TAB 3: SMART INGEST ---
with tab_upload:
    uploaded_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'avi'])
    if uploaded_file:
        save_path = os.path.join(VIDEO_FOLDER, uploaded_file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                while True:
                    chunk = uploaded_file.read(4*1024*1024)
                    if not chunk: break
                    f.write(chunk)
        
        file_hash = get_file_hash(save_path)
        c.execute("SELECT file_name FROM processed_videos WHERE file_hash = ?", (file_hash,))
        
        if c.fetchone():
            st.warning("Video already indexed.")
        else:
            if st.button("Start Identity Ingest"):
                cap = cv2.VideoCapture(save_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
                
                frame_count = 0
                processed_count = 0
                prog_bar = st.progress(0)
                status_text = st.empty()
                known_encodings_this_session = [] 
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % int(fps) == 0: 
                        prog_bar.progress(min(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1.0))
                        status_text.text(f"Analyzing frame {frame_count}...")
                        
                        mask = fgbg.apply(frame)
                        if cv2.countNonZero(mask) < (frame.shape[0]*frame.shape[1]*0.01):
                            frame_count += 1; continue
                        
                        results = yolo.predict(frame, classes=[0,1,2,3,5,7], verbose=False, device=device)
                        for r in results:
                            for box in r.boxes:
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                cls = int(box.cls[0])
                                h_img, w_img, _ = frame.shape
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w_img, x2), min(h_img, y2)
                                if (x2-x1) < 10 or (y2-y1) < 10: continue 
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0: continue
                                
                                # Face Logic
                                if cls == 0: 
                                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    h, w = rgb_crop.shape[:2]
                                    if h < 200 or w < 200: rgb_crop = cv2.resize(rgb_crop, (0, 0), fx=2, fy=2)
                                    face_locs = face_recognition.face_locations(rgb_crop, model="hog")
                                    if face_locs:
                                        encodings = face_recognition.face_encodings(rgb_crop, face_locs)
                                        for encoding in encodings:
                                            matches = face_recognition.compare_faces(known_encodings_this_session, encoding, tolerance=0.6)
                                            if not any(matches):
                                                known_encodings_this_session.append(encoding)
                                                bgr_face = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
                                                face_path = os.path.join(FACES_FOLDER, f"face_{file_hash[:8]}_{frame_count}.jpg")
                                                cv2.imwrite(face_path, bgr_face)
                                                c.execute("INSERT INTO faces (video_hash, video_name, timestamp, image_path, encoding) VALUES (?, ?, ?, ?, ?)",
                                                          (file_hash, uploaded_file.name, frame_count/fps, face_path, pickle.dumps(encoding)))

                                # BLIP Logic
                                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                inputs = blip_processor(pil_img, return_tensors="pt").to(device)
                                out = blip_model.generate(**inputs, max_new_tokens=20)
                                cap_text = blip_processor.decode(out[0], skip_special_tokens=True)
                                full_caption = f"{yolo.names[cls]}: {cap_text}"
                                vector = embedder.encode(full_caption)
                                c.execute("INSERT INTO logs (video_hash, video_name, timestamp, caption, embedding) VALUES (?, ?, ?, ?, ?)",
                                          (file_hash, uploaded_file.name, frame_count/fps, full_caption, vector.astype('float32').tobytes()))
                                processed_count += 1
                    frame_count += 1
                
                cap.release()
                import datetime
                c.execute("INSERT INTO processed_videos VALUES (?, ?, ?)", (file_hash, uploaded_file.name, str(datetime.datetime.now())))
                conn.commit()
                vector_index = rebuild_faiss_index()
                st.success(f"✅ Ingest Complete! Indexed {processed_count} events.")
                st.rerun()

# --- TAB 4: MANAGE ---
with tab_manage:
    st.header("System Management")
    
    # SECTION 1: Evidence Clips (Delete Outputs)
    st.subheader("📁 Generated Evidence Clips")
    clips = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp4")]
    
    if not clips:
        st.info("No evidence clips generated yet.")
    else:
        for clip in clips:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(clip)
            with col2:
                if st.button("Delete", key=f"del_clip_{clip}"):
                    try:
                        os.remove(os.path.join(OUTPUT_FOLDER, clip))
                        st.success(f"Deleted {clip}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    st.divider()

    # SECTION 2: Ingested Videos (Delete Raw Inputs)
    st.subheader("📼 Ingested Source Videos")
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        st.info("No source videos found.")
    else:
        for video in videos:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(video)
            with col2:
                if st.button("Delete", key=f"del_vid_{video}"):
                    try:
                        # 1. Delete File
                        file_path = os.path.join(VIDEO_FOLDER, video)
                        os.remove(file_path)
                        
                        # 2. Clean Database (Optional but recommended)
                        # We calculate hash to find DB entry (this is slow for big files, so maybe skip for now or store hash mapping)
                        # Ideally, we just delete file. To clean DB, user should use Factory Reset or we implement a complex lookup.
                        
                        st.success(f"Deleted {video}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    # SECTION 3: Nuclear Option
    st.subheader("⚠️ Factory Reset")
    st.write("This will delete the Database, Faces, Logs, Evidence Clips, AND the uploaded Source Videos.")
    
    if st.button("☢️ RESET EVERYTHING (Clean Start)", type="primary"):
        try:
            # 1. Clear Database Tables
            c.execute("DELETE FROM logs")
            c.execute("DELETE FROM processed_videos")
            c.execute("DELETE FROM faces")
            conn.commit()
            c.execute("VACUUM")
            
            # 2. Delete Files
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            
            # Helper to clear a folder
            def clear_folder(folder_path):
                if os.path.exists(folder_path):
                    for f in os.listdir(folder_path):
                        try: os.remove(os.path.join(folder_path, f))
                        except: pass
            
            clear_folder(FACES_FOLDER)
            clear_folder(OUTPUT_FOLDER)
            clear_folder(VIDEO_FOLDER) # <--- Now clears the source videos too
            
            st.success("✅ System Fully Wiped! Ready for fresh start.")
            st.rerun()
        except Exception as e:
            st.error(f"Error during wipe: {e}")
