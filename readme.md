🧠🎧 Streamlit Apps for Audio & Text Intelligence

This repo contains two powerful Streamlit applications:

MP3 Audio Summarizer (Hindi to English)
RAG Chatbot with Memory (Text/MP3 Input)
Both apps use state-of-the-art models like Whisper, Ollama (Mistral), and LangChain to extract and interact with information from text or speech.

1️⃣ MP3 Audio Summarizer (Hindi to English)

🎯 Purpose:
Transcribe a Hindi customer call recording (.mp3) using Whisper, and summarize it into bullet points in English using an LLM.

🧩 Components:
Whisper: For speech-to-text transcription.
LangChain + Ollama (Mistral): To generate a concise English summary.
Streamlit: For a clean and interactive UI.
⚙️ How It Works:
Upload an .mp3 file.
Whisper transcribes Hindi speech to text.
Mistral (via Ollama) summarizes it in bullet points.
Displays:
📝 Transcription
📌 Summary
⏱️ Time taken for each step
2️⃣ RAG Chatbot with Memory (Text/MP3 Input)

🎯 Purpose:
Ask questions about uploaded text/audio with memory-aware context using RAG (Retrieval-Augmented Generation).

🧩 Components:
Whisper: Transcribes .mp3 audio to text.
LangChain Text Splitter + Chroma DB: For chunking and storing context.
Ollama (Mistral): Answers questions based on retrieved context + conversation history.
Streamlit: For interaction and display.
⚙️ How It Works:
Upload a .txt or .mp3 file.
If audio, Whisper converts it to text.
The transcript is chunked and stored in Chroma DB.
Ask questions — the app:
Retrieves relevant chunks
Uses chat history
Responds via Mistral (Ollama)
Displays:
💬 Answer
⏱️ Time breakdown
🧠 Full conversation memory
⚙️ Getting Started

🐍 1. Create a Python 3.10 Virtual Environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
📦 2. Install All Dependencies
pip install -r requirements.txt
🤖 3. Pull the Mistral Model Using Ollama
Make sure you have Ollama installed and running.

ollama pull mistral
🚀 4. Run the MP3 Summarizer App
streamlit run summarizer.py
💬 5. Run the RAG Chatbot App
streamlit run rag.py
💡 Example Use Cases

Sales Call Analysis
Customer Support Summary
Compliance Reviews
Interview Transcripts
Voice-to-Insight Pipelines
📝 Notes

Hindi language support is powered by Whisper’s multilingual capabilities.
Vector DB (chroma_db/) persists embeddings unless manually cleared.
Time taken for transcription, retrieval, and generation is shown for transparency.