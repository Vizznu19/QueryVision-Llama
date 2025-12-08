import streamlit as st
import os
import cv2
import pickle
import subprocess
import ollama  # <--- NEW: The Brain
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- PAGE CONFIG ---
st.set_page_config(page_title="QueryVision AI (LLM)", page_icon="🧠", layout="wide")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.pkl")
VIDEO_FOLDER = os.path.join(BASE_DIR, "project_data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "search_output")

# Ensure directories exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR

# --- 🧠 LOAD VISION MODELS (CACHED) ---
@st.cache_resource
def load_models():
    st.sidebar.text("⏳ Loading Vision Models...")
    
    # 1. Load YOLO (Object Detection)
    yolo = YOLO('yolov8n.pt')
    
    # 2. Load BLIP (Captioning)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR).to("cpu")
    
    st.sidebar.success("✅ Vision Models Ready!")
    return yolo, processor, model

# Load global models
yolo, blip_processor, blip_model = load_models()

# --- HELPER: CUT VIDEO CLIP ---
def cut_clip(video_name, timestamp, duration=10):
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    start_time = max(0, timestamp - 5)
    clip_filename = f"clip_{video_name}_{int(timestamp)}s.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, clip_filename)
    
    ffmpeg_cmd = "ffmpeg"
    if os.path.exists(os.path.join(BASE_DIR, "ffmpeg.exe")):
        ffmpeg_cmd = os.path.join(BASE_DIR, "ffmpeg.exe")
    
    command = [
        ffmpeg_cmd, "-ss", str(start_time), "-i", video_path,
        "-t", str(duration), "-c", "copy", "-y", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# --- UI LAYOUT ---
st.title("🧠 QueryVision: AI Forensic Analyst")
st.markdown("### Powered by Llama 3.2")
st.markdown("---")

# Sidebar for Stats
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'rb') as f:
        meta_data = pickle.load(f)
    st.sidebar.metric("Log Entries", len(meta_data))
    
    # Debug: Show logs
    with st.sidebar.expander("View Raw Logs"):
        for entry in meta_data[:10]:
            st.text(f"[{int(entry['timestamp'])}s] {entry['caption']}")
    st.sidebar.markdown("---")
else:
    st.sidebar.warning("No Logs Found")

# TABS
tab_search, tab_upload = st.tabs(["🕵️ Ask the Analyst", "📂 Ingest Video"])

# --- TAB 1: LLM ANALYSIS ---
with tab_search:
    query = st.text_area("Ask a complex question about the footage:", 
                         placeholder="e.g. 'Did anyone enter with a red bag? Describe their actions.'")
    
    if st.button("Analyze Footage", type="primary"):
        if not os.path.exists(METADATA_FILE):
            st.error("No video logs found. Please ingest a video first.")
        else:
            with open(METADATA_FILE, 'rb') as f:
                logs = pickle.load(f)
            
            # 1. Prepare the Context for LLM
            # We convert the list of dictionaries into a readable text log
            context_log = ""
            for log in logs:
                timestamp = int(log['timestamp'])
                caption = log['caption']
                video = log['video_file']
                context_log += f"- At {timestamp} seconds in {video}: {caption}\n"
            
            # 2. The Prompt
            system_prompt = (
                "You are an expert video forensic analyst. You have access to the following CCTV logs. "
                "Your job is to answer the user's question based ONLY on these logs. "
                "If you find a match, you MUST specify the exact timestamp. "
                "Format your answer as a concise report."
            )
            
            user_prompt = f"LOGS:\n{context_log}\n\nUSER QUESTION: {query}"
            
            with st.spinner("🤖 Llama 3 is reading the logs..."):
                try:
                    # Call Ollama
                    response = ollama.chat(model='llama3.2', messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ])
                    
                    answer = response['message']['content']
                    st.success("Analysis Complete")
                    st.markdown(answer)
                    
                    # 3. Simple Keyword Match to Auto-Show Video (Bonus)
                    # We try to guess the best timestamp to show based on the LLM's text
                    # (This is a simple heuristic: find the first number mentioned in the answer)
                    import re
                    match = re.search(r'\b(\d+)\s*seconds?', answer)
                    if match:
                        best_time = int(match.group(1))
                        best_video = logs[0]['video_file'] # Default to first video
                        
                        st.divider()
                        st.subheader(f"🎥 Evidence Clip (Jump to ~{best_time}s)")
                        clip_path = cut_clip(best_video, best_time)
                        if os.path.exists(clip_path):
                            st.video(clip_path)
                    
                except Exception as e:
                    st.error(f"Ollama Error: {e}. Is Ollama running?")

# --- TAB 2: UPLOAD & PROCESS ---
with tab_upload:
    uploaded_file = st.file_uploader("Upload CCTV Footage", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        save_path = os.path.join(VIDEO_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        
        if st.button("Generate Logs (No Vectors)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(save_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * 2) 
            
            new_metadata = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_count % skip_frames == 0:
                    prog = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(prog)
                    status_text.text(f"Analyzing frame {frame_count}...")
                    
                    # YOLO Detect
                    results = yolo.predict(frame, classes=[0, 1, 2, 3, 5, 7], verbose=False, device='cpu')
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            h, w, _ = frame.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if (x2-x1) < 30 or (y2-y1) < 30: continue
                            
                            crop = frame[y1:y2, x1:x2]
                            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            
                            # BLIP Caption
                            inputs = blip_processor(pil_image, return_tensors="pt").to("cpu")
                            out = blip_model.generate(**inputs, max_new_tokens=20)
                            caption = blip_processor.decode(out[0], skip_special_tokens=True)
                            
                            cls_id = int(box.cls[0])
                            class_name = yolo.names[cls_id]
                            full_caption = f"{class_name}: {caption}"
                            
                            # Just Save Text! No Vectors!
                            new_metadata.append({
                                "video_file": uploaded_file.name,
                                "timestamp": frame_count / fps,
                                "caption": full_caption
                            })
                            
                frame_count += 1
            
            cap.release()
            
            # Save Metadata
            with open(METADATA_FILE, 'wb') as f:
                pickle.dump(new_metadata, f)
            
            progress_bar.progress(1.0)
            status_text.success(f"✅ Logs Generated! {len(new_metadata)} events recorded.")