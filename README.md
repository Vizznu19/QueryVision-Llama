# 👁️ QueryVision: AI Forensic Video Analyst

QueryVision is a local, privacy-focused video search engine. It allows you to upload CCTV footage and ask complex natural language questions like *"Did anyone steal the red bag?"* or *"Describe the sequence of events at the door."*

It uses a "Hybrid AI" architecture:
1.  **Vision Layer:** YOLOv8 (Objects) + BLIP (Captions) to "watch" the video.
2.  **Reasoning Layer:** Llama 3.2 (via Ollama) to analyze the logs and answer questions.
3.  **Interface:** A clean Streamlit Dashboard.

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
    cd query-vision-main
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
    pip install ollama
    ```

---

## ▶️ How to Run

1.  Create a folder 'project_data' and uplaod the input video.
2.  **Start Ollama:** Make sure the Ollama app is running in your background (taskbar).
3.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```
4.  **Usage:**
    * Go to the **"Ingest Video"** tab -> Upload a video -> Click "Generate Logs".
    * Go to the **"Ask the Analyst"** tab -> Type your question (e.g., *"Find the red car"*).
    * The system will answer and auto-play the evidence clip.

---

## 🧩 Troubleshooting

* **Error: `FileNotFoundError: ffmpeg`**
    * You forgot to put `ffmpeg.exe` in the root folder. Download it and drop it in.
* **Error: `Ollama connection failed`**
    * Open the Ollama app on your computer. It needs to be running to answer questions.
* **Blank Screen on Launch**
    * The first run takes time to download the YOLO/BLIP models. Check your terminal for download progress.
