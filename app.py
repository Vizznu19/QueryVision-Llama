import streamlit as st
import os
import cv2
import re
import time
import datetime
import hashlib
from PIL import Image
import ollama
from core import (
    init_db, get_db_connection, load_models, load_faiss_index, 
    rebuild_faiss_index, get_file_hash, search_logs, cut_clip,
    VIDEO_FOLDER, OUTPUT_FOLDER, DB_FILE, FAISS_INDEX_FILE
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="QueryVision: AI Survelliance System", page_icon="👁️", layout="wide")

# --- INITIALIZATION ---
conn = init_db()

@st.cache_resource
def get_cached_models():
    yolo, processor, model, embedder, device = load_models()
    st.sidebar.info(f"🚀 AI Hardware: {device.upper()}")
    return yolo, processor, model, embedder, device

yolo, blip_processor, blip_model, embedder, device = get_cached_models()

@st.cache_resource
def get_vector_index():
    return load_faiss_index(conn)

vector_index = get_vector_index()

# --- UI LAYOUT ---
st.title("QueryVision : AI Survelliance System")

c = conn.cursor()
log_count = c.execute("SELECT COUNT(*) FROM logs").fetchone()[0]

st.sidebar.metric("Events Tracked", log_count)

# REMOVED FACE GALLERY TAB
tab_search, tab_upload, tab_manage = st.tabs(["🕵️ Search Analyst", "📂 Smart Ingest", "🗑️ Manage"])

# --- TAB 1: SEARCH ---
with tab_search:
    query = st.text_area("Type your query here:", placeholder="e.g. Find the man in the blue shirt.")
    if st.button("Run Analysis", type="primary"):
        if log_count == 0:
            st.error("Database empty.")
        else:
            with st.spinner("Searching Vector Database..."):
                # ---- VECTOR SEARCH ----
                scored_logs = search_logs(query, embedder, vector_index, conn, limit=10)
                relevant_logs = [row for _, row in scored_logs]
                
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
                    st.markdown(f"**🤖 Analyst Conclusion:**\n> {answer}")
                    time_matches = re.findall(r'\[(\d+):(\d+)\]', answer)

                    if time_matches:
                        minutes, seconds = map(int, time_matches[-1])
                        total_seconds = (minutes * 60) + seconds

    # Find best matching video based on timestamp
                        best_video = None
                        closest_diff = 9999

                        for video, ts, cap in relevant_logs:
                            diff = abs(ts - total_seconds)
                            if diff < closest_diff:
                                closest_diff = diff
                                best_video = video

                        if best_video:
                            st.divider()
                            st.subheader(f"🎥 Evidence Clip ({minutes:02d}:{seconds:02d})")

                            clip_path = cut_clip(best_video, total_seconds)

                            if clip_path and os.path.exists(clip_path):
                                st.video(clip_path)
                            else:
                                st.warning("Clip could not be generated.")
                    else:
                        st.warning("No valid timestamp detected in Llama response.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: SMART INGEST (FILE & LIVE RTSP) ---
with tab_upload:
    # --- 1. UPLOAD SECTION ---
    st.subheader("📂 Upload Static Footage")
    uploaded_file = st.file_uploader("Drop CCTV Files", type=['mp4', 'avi'])
    
    # --- 2. PROCESSING LOGIC (FILE MODE) - MOVED HERE AS REQUESTED ---
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
        
        is_indexed = c.fetchone()
        if is_indexed:
            st.info(f"ℹ️ Content from '{uploaded_file.name}' is already in the database.")
            ingest_label = "🔄 Re-Ingest & Update Intelligence"
        else:
            ingest_label = "🚀 Start Turbo Ingest (File Mode)"

        if st.button(ingest_label, type="primary", use_container_width=True):
                cap = cv2.VideoCapture(save_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                
                # This analyzes 3 frames EVERY second (a great balance of speed and accuracy)
                skip_rate = int(fps / 3) 
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
                
                frame_count = 0
                processed_count = 0
                prog_bar = st.progress(0)
                
                status_text = st.empty()
                
                # --- OPEN TEXT LOG FILE FOR BACKGROUND LOGGING ---
                safe_log_name = uploaded_file.name.replace(" ", "_")
                log_file_path = os.path.join(OUTPUT_FOLDER, f"captions_log_{safe_log_name}.txt")
                log_file = open(log_file_path, "w", encoding="utf-8")
                log_file.write(f"--- AI Captioning Log for {uploaded_file.name} ---\n\n")
                # -------------------------------------------------
                
                track_history = {} 
                RE_ANALYZE_INTERVAL = int(fps * 10) 
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_count % skip_rate != 0:
                        frame_count += 1
                        continue

                    if total_frames > 0:
                        progress = frame_count / total_frames
                        prog_bar.progress(min(progress, 1.0))
                    status_text.text(f"Scanning... {processed_count} Events Logged to Database")
                    
                    mask = fgbg.apply(frame)
                    if cv2.countNonZero(mask) < (frame.shape[0]*frame.shape[1]*0.015):
                        frame_count += 1; continue
                    
                    # We added conf=0.45 (Must be 45% sure) and classes=[0,2,3,5,7] (Only Person & Vehicles)
                    results = yolo.track(frame, persist=True, verbose=False, device=device, conf=0.55, classes=[0, 2, 3, 5, 7])
                    
                    for r in results:
                        if r.boxes.id is None: continue 
                        
                        boxes = r.boxes.xyxy.cpu().numpy()
                        ids = r.boxes.id.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy()
                        
                        for box, track_id, cls in zip(boxes, ids, clss):
                            track_id = int(track_id)
                            cls = int(cls)
                            
                            last_seen = track_history.get(track_id, -99999)
                            if (frame_count - last_seen) < RE_ANALYZE_INTERVAL:
                                continue 
                            
                            x1,y1,x2,y2 = map(int, box)
                            h_img, w_img, _ = frame.shape
                            
                            width = x2 - x1
                            height = y2 - y1
                            
                            if width < 50 or height < 50: continue 

                            # --- ACCURACY FIX 1: CONTEXT PADDING ---
                            pad_x = int(width * 0.15)
                            pad_y = int(height * 0.15)
                            
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(w_img, x2 + pad_x)
                            y2 = min(h_img, y2 + pad_y)
                            # ---------------------------------------

                            track_history[track_id] = frame_count
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0: continue
                            
                            # (FACE DETECTION LOGIC REMOVED HERE)

                            # --- ACCURACY FIX 2 & 3: UPSCALING & GUIDED PROMPTING ---
                            rgb_crop_blip = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            bh, bw = rgb_crop_blip.shape[:2]
                            
                            if bh < 224 or bw < 224: 
                                rgb_crop_blip = cv2.resize(rgb_crop_blip, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                            
                            pil_img = Image.fromarray(rgb_crop_blip)
                            class_name = yolo.names[cls]
                            
                            if cls == 0:
                                prompt_text = "Describe the person's clothing, colors, and visible actions in detail."
                            elif cls in [2,3,5,7]:
                                prompt_text = f"Describe the {class_name}, its color, and its movement."
                            else:
                                prompt_text = f"Describe the {class_name} clearly."

                            inputs = blip_processor(pil_img, text=prompt_text, return_tensors="pt").to(device)
                            out = blip_model.generate(**inputs, max_new_tokens=20)
                            cap_text = blip_processor.decode(out[0], skip_special_tokens=True)
                            
                            full_caption = f"{class_name}: {cap_text.strip()}"
                            # --------------------------------------------------------

                            vector = embedder.encode(full_caption, normalize_embeddings=True)
                            c.execute("INSERT INTO logs (video_hash, video_name, timestamp, caption, embedding) VALUES (?, ?, ?, ?, ?)",
                                      (file_hash, uploaded_file.name, frame_count/fps, full_caption, vector.astype('float32').tobytes()))
                            processed_count += 1
                            
                            # --- WRITE CAPTION TO LOG FILE ---
                            m, s = divmod(int(frame_count/fps), 60)
                            log_file.write(f"[Frame {frame_count} | {m:02d}:{s:02d}] {full_caption}\n")
                            log_file.flush()
                            # --------------------------------------

                    frame_count += 1
                
                cap.release()
                log_file.close() 
                
                import datetime
                c.execute("INSERT OR IGNORE INTO processed_videos VALUES (?, ?, ?)", (file_hash, uploaded_file.name, str(datetime.datetime.now())))
                conn.commit()
                st.cache_resource.clear()
                vector_index = rebuild_faiss_index(conn)
                st.success(f"✅ Ingest Complete! Processed {processed_count} significant events. Reference log saved in 'search_output' folder.")
                st.rerun()

    st.divider()
    # --- 3. LIVE SURVEILLANCE SECTION ---
    st.subheader("🖥️ Live RTSP Analysis")
    
    # --- SESSION STATE & PRESET LOGIC ---
    if 'live_ingest' not in st.session_state:
        st.session_state.live_ingest = False
    
    if 'live_prev' not in st.session_state:
        st.session_state.live_prev = False
        
    if 'rtsp_url_input' not in st.session_state:
        st.session_state.rtsp_url_input = ""
    
    # Sidebar persistence (Show even when stopped)
    st.sidebar.divider()
    st.sidebar.subheader("🔔 Live Event Log")
    sidebar_evt_log = st.sidebar.empty()
    sidebar_log_text = ""
    
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        rtsp_url = st.text_input("RTSP URL (or Camera IP)", key="rtsp_url_input", placeholder="rtsp://admin:password@192.168.1.1:554/live")
    with col_btn:
        st.write("") # Spacer
        if not st.session_state.live_ingest:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                if rtsp_url:
                    st.session_state.live_ingest = True; st.session_state.live_prev = False; st.rerun()
                else: st.error("Please enter an RTSP URL.")
        else:
            if st.button("🛑 Stop Analysis", type="primary", use_container_width=True):
                st.session_state.live_ingest = False; st.rerun()

    if not st.session_state.live_ingest:
        if not st.session_state.live_prev:
            if st.button("📺 Preview Stream", use_container_width=True):
                if rtsp_url:
                    st.session_state.live_prev = True; st.session_state.live_ingest = False; st.rerun()
                else: st.error("Please enter an RTSP URL.")
        else:
            if st.button("❌ Close Preview", use_container_width=True):
                st.session_state.live_prev = False; st.rerun()

    # --- SHARED RTSP STABILITY SETTINGS ---
    # Force UDP Transport for better Wi-Fi stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    # --- LIVE PREVIEW MODE (NO AI) ---
    if st.session_state.live_prev:
        st.info(f"Connecting to live feed...")
        src = 0 if rtsp_url == "0" else rtsp_url
        # Use FFMPEG backend and lower buffer for RTSP speed
        cap_p = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap_p.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap_p.isOpened():
            st.error("❌ Connection failed. Check the URL and ensure your phone's 'Start Server' is ON.")
            st.session_state.live_prev = False
        else:
            prev_mon = st.empty()
            prev_mon.warning("⏳ Waiting for video signal...")
            while st.session_state.live_prev:
                ret, frame = cap_p.read()
                if not ret: break
                prev_mon.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live Preview", use_container_width=True)
                time.sleep(0.01)
            cap_p.release()

    # --- LIVE RTSP PROCESSING LOOP ---
    if st.session_state.live_ingest:
        st.info(f"📡 Initializing AI Intelligence Feed...")
        src = 0 if rtsp_url == "0" else rtsp_url
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("❌ Failed to connect to source. Check settings.")
            st.session_state.live_ingest = False
        else:
            # Create a unique session ID for this live stream
            session_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"LIVE_{session_start}"
            session_hash = "LIVE_" + hashlib.sha256(session_name.encode()).hexdigest()[:10]
            
            st.success(f"🎬 Active Session: {session_name}")
            
            # --- MONITOR WINDOWS ---
            status_placeholder = st.sidebar.empty()
            metric_placeholder = st.sidebar.empty()
            
            col_live, col_log = st.columns([2, 1])
            with col_live:
                live_monitor = st.empty()
                live_monitor.warning("⏳ Waiting for first AI frame...")
            with col_log:
                event_log = st.empty()
                log_text = "🔔 Live Events:\n\n"

            processed_count = 0
            track_history = {}
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            skip_rate = 5 # Improved speed: Process slightly more frames
            frame_count = 0
            RE_ANALYZE_INTERVAL = int(fps * 5) # 5 seconds

            error_streak = 0
            while st.session_state.live_ingest:
                status_placeholder.info("🕵️ AI is scanning live feed...")
                metric_placeholder.metric("Events Logged (This Session)", processed_count)
                
                # --- FLUSH BUFFER (Essential for real-time stability) ---
                # We grab 5 frames and only retrieve the last one to ensure no lag
                for _ in range(5): cap.grab()
                ret, frame = cap.retrieve()
                
                if not ret:
                    error_streak += 1
                    if error_streak > 15: # Attempt to survive noise
                        st.warning("⚠️ Signal weak. Reconnecting...")
                        cap.release(); cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if error_streak > 50: # Hard stop after 50 failures
                            st.error("❌ Stream lost.")
                            break
                    continue
                
                error_streak = 0
                frame_count += 1
                
                # --- YOLO INTELLIGENCE ---
                # We analyze every N frames for heavy lifting, but we can show boxes more often if needed.
                # For CPU, we keep it at every 10 frames as per user setup.
                if frame_count % 10 == 0:
                    # 'predict' is more reliable for live streams than 'track'
                    results = yolo.predict(frame, verbose=False, device=device, conf=0.35)
                    
                    # --- FIX: Create a clean copy for the AI to analyze without boxes ---
                    clean_frame = frame.copy() 
                    
                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy()
                        
                        for box, cls in zip(boxes, clss):
                            cls = int(cls)
                            class_name = yolo.names[cls]
                            
                            # DRAWING (Visible to user only)
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name.upper()}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Throttling
                            last_seen = track_history.get(class_name, -99999)
                            if (frame_count - last_seen) < RE_ANALYZE_INTERVAL:
                                continue 
                            
                            track_history[class_name] = frame_count
                            
                            # BLIP CAPTIONING (Crop from CLEAN frame)
                            status_placeholder.info(f"🔍 AI Analyzing {class_name}...")
                            
                            h_img, w_img, _ = clean_frame.shape
                            pad_x, pad_y = int((x2-x1)*0.15), int((y2-y1)*0.15)
                            px1, py1 = max(0, x1-pad_x), max(0, y1-pad_y)
                            px2, py2 = min(w_img, x2+pad_x), min(h_img, y2+pad_y)
                            
                            pil_img = Image.fromarray(cv2.cvtColor(clean_frame[py1:py2, px1:px2], cv2.COLOR_BGR2RGB))
                            
                            status_placeholder.info(f"🔍 Analyzing {class_name}...")
                            prompt_text = f"Describe the {class_name} in detail."
                            inputs = blip_processor(pil_img, text=prompt_text, return_tensors="pt").to(device)
                            out = blip_model.generate(**inputs, max_new_tokens=20)
                            cap_text = blip_processor.decode(out[0], skip_special_tokens=True)
                            
                            full_caption = f"{class_name}: {cap_text.strip()}"
                            vector = embedder.encode(full_caption, normalize_embeddings=True)
                            
                            # LOG TO DATABASE (MANDATORY)
                            curr_ts = time.time()
                            c.execute("INSERT INTO logs (video_hash, video_name, timestamp, caption, embedding) VALUES (?, ?, ?, ?, ?)",
                                      (session_hash, session_name, curr_ts, full_caption, vector.astype('float32').tobytes()))
                            conn.commit()
                            processed_count += 1
                            
                            # UPDATE UI LOGS
                            log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {full_caption}\n"
                            log_text = log_entry + log_text
                            sidebar_log_text = log_entry + sidebar_log_text
                            event_log.markdown(log_text)
                            sidebar_evt_log.text(sidebar_log_text)
                
                # Update Dashboard Monitor with Boxes
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                live_monitor.image(rgb_frame, caption="Live Intelligence Feed (Active Analysis)", use_container_width=True)
                
            cap.release()
            # Register session as a "video" so it shows in search/manage
            c.execute("INSERT OR IGNORE INTO processed_videos VALUES (?, ?, ?)", 
                      (session_hash, session_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            rebuild_faiss_index(conn)
            st.cache_resource.clear()
            st.session_state.live_ingest = False
            st.rerun()

    # End of Ingest Section

# --- TAB 3: MANAGE ---
with tab_manage:
    st.header("System Management")
    
    st.subheader("📁 Generated Evidence Clips & Logs")
    clips = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith((".mp4", ".txt"))]
    
    if not clips:
        st.info("No evidence clips or logs generated yet.")
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
                        file_path = os.path.join(VIDEO_FOLDER, video)
                        os.remove(file_path)
                        st.success(f"Deleted {video}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    st.subheader("Factory Reset")
    st.write("This will delete the Database, Logs, Evidence Clips, AND the uploaded Source Videos.")
    
    if st.button("RESET EVERYTHING (Clean Start)", type="primary"):
        try:
            c.execute("DELETE FROM logs")
            c.execute("DELETE FROM processed_videos")
            # REMOVED: c.execute("DELETE FROM faces")
            conn.commit()
            c.execute("VACUUM")
            
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            
            def clear_folder(folder_path):
                if os.path.exists(folder_path):
                    for f in os.listdir(folder_path):
                        try: os.remove(os.path.join(folder_path, f))
                        except: pass
            
            # REMOVED: clear_folder(FACES_FOLDER)
            clear_folder(OUTPUT_FOLDER)
            clear_folder(VIDEO_FOLDER) 
            
            st.success("✅ System Fully Wiped! Ready for fresh start.")
            st.rerun()
        except Exception as e:
            st.error(f"Error during wipe: {e}")