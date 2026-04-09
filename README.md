# 👁️ QueryVision: AI Surveillance & Forensic Analyst

QueryVision is a powerful, **100% local**, and privacy-focused AI surveillance platform. It transforms standard CCTV footage or live phone streams (RTSP) into a searchable intelligence database.

> *"Did anyone touch my laptop while I was away?"* — QueryVision doesn't just watch; it understands and reasons.

---

## 🌟 Key Features

### 1. 🚀 Turbo Ingest (Forensic Mode)
Upload massive CCTV files and index them in minutes. 
- **Motion Filtering:** Automatically skips "empty" footage to save CPU/GPU power.
- **High-Detail Crops:** Automatically "zooms in" on objects for clear identification.

### 2. 🖥️ Live RTSP Analysis (Surveillance Mode)
Connect your phone or IP camera via RTSP for real-time intelligence.
- **Visual Overlays:** Real-time green bounding boxes and tracking IDs.
- **Surveillance Logs:** Instant text logs of every person or object detected.

### 3. 🧠 Hybrid AI Brain
- **Detection:** YOLOv8 (identifies people, bags, laptops, vehicles).
- **Description:** BLIP (generates natural language captions for every detection).
- **Memory:** Sentence-Transformers + FAISS (stores meanings as 384D vectors).
- **Reasoning:** Llama 3 (Ollama) acts as your expert Forensic Analyst.

### 4. 🌐 MCP Server (Model Context Protocol) 
QueryVision now includes an **MCP Server**, allowing you to connect **Claude Desktop** directly to your cameras. You can "talk" to your videos from anywhere.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama:** Install from [ollama.com](https://ollama.com/) and run `ollama pull llama3.2`.
- **FFmpeg:** Place `ffmpeg.exe` in the root directory for evidence clipping.

### 2. Installation
```bash
git clone https://github.com/Vizznu19/QueryVision-Llama.git
cd QueryVision-Llama
pip install -r requirements.txt
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

### 4. Optional: Start MCP Server
```bash
python mcp_server.py
```

---

## 🛠️ Architecture

QueryVision uses a **Vector-Reasoning** architecture:
1. **YOLO** detects an object.
2. **BLIP** describes the object in detail.
3. **SentenceBERT** converts the description to a math vector.
4. **FAISS** searches the vectors.
5. **Llama 3** reads the search results and explains them to you.

---

## 🤝 Contributing
Feel free to fork this project and add new detection classes or improved VLM models. Designed for the **Advanced AI Surveillance Expo**.
