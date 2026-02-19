👁️ QueryVision Pro: AI Forensic Video Analyst
QueryVision Pro is a local, privacy-focused video search engine. It allows you to upload CCTV or highway footage and ask complex natural language questions like "Find the suspect wearing a red shirt" or "Find the white commercial truck."

🎯 Key Features
🔍 Semantic Hybrid Search: Combines FAISS vector similarity search with LLM reasoning for intelligent, time-accurate video querying.

⚡ Turbo Ingestion Pipeline: Uses a multi-layered filtering funnel (hash checking, temporal motion masking, and ID debouncing) to process heavy video rapidly.

🎥 Automated Context Captions: Uses YOLOv8 for spatial object tracking and BLIP with context-padding for highly accurate forensic descriptions.

💾 Dual Database System: Uses SQLite for strict metadata logging and FAISS for high-speed mathematical vector retrieval.

🎬 Evidence Extraction: Automatically snips and generates .mp4 evidence clips at the exact moment a match is found.

⚙️ System Management: A polished, enterprise-grade UI to manage databases, delete footage, or perform a secure factory reset.

🏗️ Architecture
The system utilizes a "Hybrid AI" pipeline:

Vision Layer: YOLOv8 (Object Tracking) + BLIP (Image Captioning) to analyze and describe video frames.

Search Layer: Sentence Transformers (Embeddings) + FAISS (Vector Database) for semantic text-to-video matching.

Reasoning Layer: Llama 3.2 (via Ollama) to logically analyze the retrieved logs and pinpoint exact timestamps.

Interface: A clean, light-themed Streamlit Dashboard with 3 primary forensic tabs.

🚀 Prerequisites
Before running the project, you need these installed on your system:

Python 3.10+

Ollama (The AI Brain)

Download from ollama.com.

Install it and run ollama pull llama3.2 in your system terminal.

FFmpeg (The Video Cutter)

Download the "ffmpeg-essentials" build from gyan.dev.

Extract ffmpeg.exe and place it directly inside this project folder (next to app.py).

🛠️ Installation
Clone the repository:

Bash

git clone <YOUR_REPO_URL_HERE>
cd query-vision-llama
Create a Virtual Environment (Recommended):

Bash

python -m venv .venv

# Windows:
.\.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
Install Dependencies:

Bash

pip install streamlit opencv-python sqlite3-api ollama torch torchvision faiss-cpu numpy sentence-transformers transformers pillow ultralytics
▶️ How to Run
Start Ollama: Make sure the Ollama app is running in your system background.

Launch the Dashboard:

Bash

streamlit run app.py
(Alternatively, use python ingest.py for headless, background processing of massive highway datasets).

Usage Workflow:

Step 1: Smart Ingest (📂 Tab)

Upload your CCTV footage (MP4 or AVI format).

Click "Start Turbo Ingest Pipeline".

The system applies the filtering funnel:

Downsamples to ~3 frames per second (forensic sweet spot).

Drops static frames via motion detection.

Tracks distinct objects to avoid redundant processing.

Generates enriched AI captions and logs them to the database.

Duplicate files are automatically detected and bypassed.

Step 2: Search Analyst (🕵️ Tab)

Type your natural language query (e.g., "Find the man in the blue shirt").

Click "Run Forensic Analysis".

The system will:

Convert your query to vector embeddings.

Search for similar visual events in the FAISS index.

Use Llama 3.2 to deduce the absolute best match.

Extract the timestamp and generate an MP4 evidence clip right on your screen.

Step 3: System Manage (⚙️ Tab)

View and delete generated evidence clips and text logs.

View and delete ingested source videos.

Factory Reset: Securely wipe the SQLite database, FAISS index, and all media folders for a clean slate.

📊 How the Optimization Funnel Works
To process heavy Vision-Language models on standard hardware, the pipeline uses strict optimizations:

File Hashing: SHA-256 checks prevent re-indexing the same video twice.

Temporal Masking: Drops 90% of frames. Only active frames (motion > 1.5%) are passed to YOLO.

Spatial Filtering: YOLO is restricted to forensic classes (Persons, Cars, Trucks). Objects under 50x50 pixels are discarded to prevent AI hallucinations.

ID Debouncing: YOLO assigns unique track IDs. If a person stands still for 5 minutes, the system enforces a cooldown, only sending them to BLIP once every 10 seconds.

Context Padding: YOLO bounding boxes are mathematically padded by 15% to give BLIP background context for more accurate descriptive captions.

🗂️ Project Structure
Plaintext

query-vision-llama/
├── app.py                  # Main Streamlit forensic dashboard
├── ingest.py               # Standalone script for CLI batch ingestion
├── ffmpeg.exe              # FFmpeg executable for video clipping
├── queryvision.db          # SQLite database (Auto-generated)
├── faiss_store.index       # FAISS vector index (Auto-generated)
├── project_data/           # Uploaded source videos directory
├── search_output/          # Generated evidence clips & text logs
└── hf_cache/               # Local cache for heavy Hugging Face models
🧩 Troubleshooting
Error: FileNotFoundError: ffmpeg

You forgot to put ffmpeg.exe in the root folder. Download it and drop it in.

Error: Ollama connection failed

Open the Ollama app on your computer. It needs to be running to answer questions. Ensure you have run ollama pull llama3.2.

System Freezing During Ingestion

Video processing is RAM-heavy. Close background applications. The system automatically utilizes CUDA if an NVIDIA GPU is present.

Video Already Indexed Warning

The system strictly prevents duplicate processing. To force a reprocess, use the Factory Reset in the Manage tab.

Blank Screen on First Launch

The very first run takes time to download the YOLO, BLIP, and Sentence Transformer models. Check your terminal for the download progress bars.
