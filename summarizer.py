import streamlit as st
import whisper
import os
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time  # Import time module for measuring elapsed time

# Load Whisper model once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Transcribe audio using Whisper
def transcribe_audio(audio_file_path):
    model = load_whisper_model()
    result = model.transcribe(audio_file_path)
    return result['text']

# Set up Ollama + prompt
def summarize_transcription(transcription):
    prompt_template = """
    Summarize the following customer lead call that's in Hindi into bullet points in English:

    {transcription}

    Summary:
    """
    prompt = PromptTemplate(input_variables=["transcription"], template=prompt_template)
    llm = Ollama(model="mistral")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(transcription)

# Streamlit UI
st.set_page_config(page_title="MP3 to Summary App", layout="centered")
st.title("🎙️ MP3 Audio Summarizer (Hindi to English)")

st.markdown("""
Upload a Hindi customer call recording (`.mp3` format). 
The app will transcribe it using Whisper and summarize it in bullet points using a language model.
""")

uploaded_audio = st.file_uploader("Upload an MP3 File", type=["mp3"])

if uploaded_audio is not None:
    with st.spinner("Saving and processing audio..."):
        # Measure time for saving the audio
        start_time = time.time()
        audio_path = os.path.join("temp_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())
        save_time = time.time() - start_time
        st.success(f"Audio uploaded. Transcribing... (Time taken: {save_time:.2f} seconds)")

        # Transcription
        with st.spinner("Transcribing audio to text..."):
            start_time = time.time()
            transcription = transcribe_audio(audio_path)
            transcribe_time = time.time() - start_time
            st.markdown("#### 📝 Transcription")
            st.text_area("Transcribed Text", transcription, height=200)
            st.success(f"Transcription complete. (Time taken: {transcribe_time:.2f} seconds)")

        # Summary
        with st.spinner("Summarizing..."):
            start_time = time.time()
            summary = summarize_transcription(transcription)
            summarize_time = time.time() - start_time
            st.markdown("#### 📌 Summary")
            st.markdown(summary.replace("-", "•"))
            st.success(f"Summary complete. (Time taken: {summarize_time:.2f} seconds)")

        # Optional cleanup
        os.remove(audio_path)
