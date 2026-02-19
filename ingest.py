import os

# --- 🛡️ CRASH GUARDS (MUST BE FIRST) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import torch
import numpy as np
import pickle
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --- STRICT THREADING LOCKS ---
cv2.setNumThreads(0)
torch.set_num_threads(1)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
os.environ['HF_HOME'] = CACHE_DIR

VIDEO_FOLDER = os.path.join(BASE_DIR, "project_data")
INDEX_FILE = os.path.join(BASE_DIR, "vector_store.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.pkl")

SKIP_SECONDS = 2

def save_index(vectors, metadata):
    """
    ⚠️ IMPORT FAISS HERE ONLY ⚠️
    """
    print("\n📦 Importing FAISS for saving...")
    import faiss  
    
    dimension = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype('float32'))
    
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ SUCCESS! Saved {len(vectors)} events to index.")

def main():
    print("--- STARTING INGESTION (HIGHWAY MODE) ---")
    
    # 1. LOAD MODELS
    print("⏳ Loading YOLO...")
    yolo = YOLO('yolov8n.pt', task='detect') 
    
    print("⏳ Loading BLIP (Color & Context Expert)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR).to("cpu")
    
    print("⏳ Loading Embedder...")
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR, device="cpu")

    all_vectors = []
    all_metadata = [] 
    
    # 2. PROCESS VIDEO
    if not os.path.exists(VIDEO_FOLDER):
        print(f"❌ Error: {VIDEO_FOLDER} does not exist.")
        return

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_name in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        print(f"\n▶️ Processing: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        frame_skip = int(fps * SKIP_SECONDS)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # YOLO Detection
            try:
                # --- 🚦 CHANGED HERE: Added Highway Vehicles ---
                # 0=Person, 1=Bicycle, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
                results = yolo.predict(frame, verbose=False, device='cpu')
                
                # OPTION: If you want EVERYTHING (Traffic lights, signs, birds), remove 'classes=...' entirely:
                # results = yolo.predict(frame, verbose=False, device='cpu')
            except Exception as e:
                print(f"❌ YOLO Error: {e}")
                break
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Decreased size limit slightly for distant cars
                    if (x2-x1) < 30 or (y2-y1) < 30: continue 

                    crop = frame[y1:y2, x1:x2]
                    
                    # Caption & Embed
                    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inputs = blip_processor(pil_image, return_tensors="pt").to("cpu")
                    out = blip_model.generate(**inputs, max_new_tokens=20)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    
                    # Append class name to caption to help search (e.g. "car: a red sedan...")
                    cls_id = int(box.cls[0])
                    class_name = yolo.names[cls_id]
                    enriched_caption = f"{class_name}: {caption}"

                    vector = text_embedder.encode(enriched_caption)
                    
                    all_vectors.append(vector)
                    all_metadata.append({
                        "video_file": video_name,
                        "timestamp": frame_count / fps,
                        "caption": enriched_caption # storing the better caption
                    })

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"   Processed {frame_count} frames...", end='\r')

        cap.release()

    # 3. SAVE
    if len(all_vectors) > 0:
        save_index(all_vectors, all_metadata)
    else:
        print("\n❌ No relevant activity found.")

if __name__ == "__main__":
    main()
