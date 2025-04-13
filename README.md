# 🧠🎧 Streamlit Apps for Audio & Text Intelligence

This repo contains two powerful Streamlit applications:

- **MP3 Audio Summarizer (Hindi to English)**
- **RAG Chatbot with Memory (Text/MP3 Input)**

Both apps use state-of-the-art models like **Whisper**, **Ollama (Mistral)**, and **LangChain** to extract and interact with information from text or speech.

---

## 1️⃣ MP3 Audio Summarizer (Hindi to English)

### 🎯 Purpose

Transcribe a Hindi customer call recording (`.mp3`) using **Whisper**, and summarize it into bullet points in **English** using an LLM.

### 🧩 Components

- **Whisper** – for speech-to-text transcription  
- **LangChain + Ollama (Mistral)** – to generate a concise English summary  
- **Streamlit** – for a clean and interactive UI

### ⚙️ How It Works

1. Upload an `.mp3` file  
2. Whisper transcribes Hindi speech to text  
3. Mistral (via Ollama) summarizes it in bullet points  
4. The app displays:
   - 📝 **Transcription**
   - 📌 **Summary**
   - ⏱️ **Time taken** for each step

---

## 2️⃣ RAG Chatbot with Memory (Text/MP3 Input)

### 🎯 Purpose

Ask questions about uploaded text/audio with memory-aware context using **RAG (Retrieval-Augmented Generation)**.

### 🧩 Components

- **Whisper** – transcribes `.mp3` audio to text  
- **LangChain Text Splitter + Chroma DB** – for chunking and storing context  
- **Ollama (Mistral)** – answers questions based on retrieved context + conversation history  
- **Streamlit** – for interaction and display

### ⚙️ How It Works

1. Upload a `.txt` or `.mp3` file  
2. If audio, Whisper converts it to text  
3. The transcript is chunked and stored in **Chroma DB**  
4. Ask questions — the app:
   - Retrieves relevant chunks
   - Uses chat history
   - Responds via **Mistral (Ollama)**
5. The app displays:
   - 💬 **Answer**
   - ⏱️ **Time breakdown**
   - 🧠 **Full conversation memory**

---

## ⚙️ Getting Started

### 🐍 1. Create a Python 3.10 Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 📦 2. Install All Dependencies
```bash
pip install -r requirements.txt
```
### 🤖 3. Pull the Mistral Model Using Ollama
Make sure Ollama is installed on you laptop and running:
```bash
ollama pull mistral
```
### 🚀 4. Run the MP3 Summarizer App
```bash
streamlit run summarizer.py
```
### 💬 5. Run the RAG Chatbot App
```bash
streamlit run rag.py
```

## 💡 Example Use Cases

1. Sales Call Analysis
2. Customer Support Summary
3. Compliance Reviews
4. Interview Transcripts
5. Voice-to-Insight Pipelines

## 📝 Notes

1. Hindi language support is powered by Whisper’s multilingual capabilities
2. Vector DB (chroma_db/) persists embeddings unless manually cleared
3. Time taken for transcription, retrieval, and generation is shown for transparency
