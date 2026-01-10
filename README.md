# 👁️ QueryVision Pro: Identity-Aware AI Forensic Video Analyst

QueryVision Pro is a local, privacy-focused video search engine with identity-aware capabilities. It allows you to upload CCTV footage and ask complex natural language questions like *"Find the man in the blue shirt"* or *"Show me all faces detected in this video."*

## 🎯 Key Features

- **🔍 Hybrid Search**: Vector similarity search + LLM reasoning for intelligent video querying
- **👥 Face Recognition**: Automatic face detection and identification with deduplication
- **🎥 Smart Ingestion**: Motion detection, duplicate prevention, and automated indexing
- **💾 Database System**: SQLite database with FAISS vector indexing for fast retrieval
- **🎬 Evidence Clips**: Automatic video clip generation at relevant timestamps
- **🗑️ System Management**: Full database and file management with factory reset option

## 🏗️ Architecture

It uses a "Hybrid AI" architecture:
1.  **Vision Layer:** YOLOv8 (Object Detection) + BLIP (Image Captioning) to analyze video frames
2.  **Identity Layer:** Face Recognition library for person identification and tracking
3.  **Search Layer:** FAISS (Vector Search) + Sentence Transformers (Embeddings) for semantic search
4.  **Reasoning Layer:** Llama 3.2 (via Ollama) to analyze logs and answer questions intelligently
5.  **Interface:** A clean Streamlit Dashboard with 4 main tabs

---

## 🚀 Prerequisites

Before running the project, you need these installed on your system:

1.  **Python 3.10+**
2.  **Ollama** (The AI Brain)
    * Download from [ollama.com](https://ollama.com/).
    * Install it and run `ollama pull llama3.2` in your system terminal.
3.  **FFmpeg** (The Video Cutter)
    * Download "ffmpeg-essentials" build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    * Extract `ffmpeg.exe` and place it **directly inside this project folder** (next to `app.py`).

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_URL_HERE>
    cd query-vision-llama
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install streamlit ollama face-recognition
    ```
    
    **Note:** On Windows, if `face-recognition` installation fails, you may need to install `cmake` and `dlib` first:
    ```bash
    pip install cmake
    pip install dlib
    pip install face-recognition
    ```

---

## ▶️ How to Run

1.  **Start Ollama:** Make sure the Ollama app is running in your background (taskbar). If you haven't pulled the model yet, run `ollama pull llama3.2` in terminal.
2.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```
3.  **Usage Workflow:**

    **Step 1: Smart Ingest (📂 Tab)**
    * Upload your CCTV footage (MP4 or AVI format)
    * Click "Start Identity Ingest"
    * The system will:
      - Process 1 frame per second
      - Filter static frames using motion detection
      - Detect objects (persons, vehicles) using YOLO
      - Generate captions using BLIP
      - Extract and identify unique faces
      - Index everything in the database
    * Progress bar shows ingestion status
    * Duplicate videos are automatically detected and skipped

    **Step 2: Search Analyst (🕵️ Tab)**
    * Type your natural language query (e.g., *"Find the person in the blue shirt"*)
    * Click "Run Analysis"
    * The system will:
      - Convert your query to vector embeddings
      - Search similar events in the FAISS index
      - Use Llama 3.2 to reason about the best match
      - Generate an evidence clip automatically
      - Display the clip at the relevant timestamp

    **Step 3: Face Gallery (👥 Tab)**
    * View all unique faces detected across all videos
    * See timestamp and source video for each face
    * Faces are automatically deduplicated (same person appears once)

    **Step 4: Manage (🗑️ Tab)**
    * View and delete generated evidence clips
    * View and delete ingested source videos
    * Factory Reset: Completely wipe all data (database, faces, videos, clips)

---

## 📊 How It Works

### Processing Pipeline

1. **Video Ingestion:**
   - Frame sampling (1 frame per second)
   - Motion detection filtering (skips static frames)
   - Object detection (YOLO: persons, vehicles)
   - Face extraction and encoding (face_recognition library)
   - Image captioning (BLIP: generates descriptions)
   - Text embedding (Sentence Transformers: converts to vectors)
   - Database storage (SQLite + FAISS index)

2. **Search Process:**
   - User query → Text embedding
   - Vector similarity search (FAISS)
   - Retrieve top matches from database
   - LLM reasoning (Llama 3.2) with context
   - Extract timestamp from LLM response
   - Generate evidence clip (FFmpeg)

3. **Face Recognition:**
   - Detects faces in person bounding boxes
   - Upscales small faces for better detection
   - Encodes faces using dlib/HOG model
   - Deduplicates faces within same video
   - Stores face images and encodings

## 🗂️ Project Structure

```
query-vision-llama/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── ffmpeg.exe             # FFmpeg executable (required)
├── yolov8n.pt             # YOLO model (auto-downloaded)
├── queryvision.db         # SQLite database (auto-created)
├── faiss_store.index      # FAISS vector index (auto-created)
├── project_data/          # Uploaded videos storage
├── project_faces/         # Extracted face images
├── search_output/         # Generated evidence clips
└── hf_cache/              # Hugging Face model cache
```

## 🧩 Troubleshooting

* **Error: `FileNotFoundError: ffmpeg`**
    * You forgot to put `ffmpeg.exe` in the root folder. Download it and drop it in.

* **Error: `Ollama connection failed`**
    * Open the Ollama app on your computer. It needs to be running to answer questions.
    * Make sure you've run `ollama pull llama3.2` to download the model.

* **Blank Screen on Launch**
    * The first run takes time to download the YOLO/BLIP/Sentence Transformer models. Check your terminal for download progress.
    * Models are cached in `hf_cache/` folder after first download.

* **Face Recognition Not Working**
    * Make sure `face-recognition` library is installed: `pip install face-recognition`
    * On Windows, you may need to install `cmake` and `dlib` first
    * Face recognition uses HOG model (CPU-based) for compatibility

* **Slow Processing**
    * Processing 1 frame per second is normal. Large videos will take time.
    * Use GPU if available (CUDA) for faster YOLO/BLIP inference
    * Motion detection helps skip static frames and speed up processing

* **Video Already Indexed Warning**
    * The system uses SHA-256 file hashing to detect duplicates
    * To reprocess, use Factory Reset in Manage tab, or manually delete from database

* **Database Errors**
    * If database gets corrupted, delete `queryvision.db` and restart
    * Use Factory Reset in Manage tab for a clean slate
