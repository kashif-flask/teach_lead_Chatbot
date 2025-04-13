import os
import streamlit as st
import whisper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time  # Import time module for measuring elapsed time

# Constants
persist_directory = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
whisper_model = whisper.load_model("base")

# Split transcript into chunks
def chunk_text_with_splitter(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)

# Upload file (TXT or MP3)
def upload_file():
    uploaded_file = st.file_uploader("Upload your transcript (.txt) or audio (.mp3)", type=["txt", "mp3"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")
            return "text", text
        elif file_extension == "mp3":
            temp_audio_path = os.path.join("temp_audio.mp3")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())
            result = whisper_model.transcribe(temp_audio_path)
            return "audio", result["text"]
    return None, None

# Create and store chunks in Chroma vector DB
def create_chromadb_index(chunks):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    return Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

# Load existing vector DB
def load_existing_vectorstore():
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# Retrieve relevant chunks for a query
def retrieve_relevant_chunks(query, vector_store, top_k=5):
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

# Format memory into string
def format_memory(history):
    return "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in history])

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["memory", "context", "query"],
    template="""
<chat_history>{memory}</chat_history>
<context>{context}</context>
Use the chat history and the context above to answer the user query. 
Only use the context and history; do not make up information.

User Query: {query}
Answer:
"""
)

# Setup LLMChain
llm = Ollama(model="mistral")
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Generate answer
def generate_answer(context, query, memory):
    return llm_chain.run({
        "context": context,
        "query": query,
        "memory": memory
    })

# RAG + memory workflow
def rag_summary_workflow(query, use_existing=False, transcript_text=None, history=None):
    start_time = time.time()  # Start the workflow timer
    
    if not use_existing and transcript_text:
        chunk_start_time = time.time()  # Timer for chunking text
        chunks = chunk_text_with_splitter(transcript_text)
        chunking_time = time.time() - chunk_start_time  # Time for chunking
        vector_store_start_time = time.time()  # Timer for creating the vector store
        vector_store = create_chromadb_index(chunks)
        vector_store_creation_time = time.time() - vector_store_start_time  # Time for vector store creation
    else:
        vector_store_start_time = time.time()  # Timer for loading the vector store
        vector_store = load_existing_vectorstore()
        vector_store_loading_time = time.time() - vector_store_start_time  # Time for vector store loading
    
    retrieve_start_time = time.time()  # Timer for retrieving relevant chunks
    relevant_context = retrieve_relevant_chunks(query, vector_store)
    retrieval_time = time.time() - retrieve_start_time  # Time for retrieval

    memory_text = format_memory(history or [])
    answer_start_time = time.time()  # Timer for generating the answer
    answer = generate_answer(" ".join(relevant_context), query, memory_text)
    answer_time = time.time() - answer_start_time  # Time for generating the answer

    if history is not None:
        history.append({"query": query, "response": answer})

    total_time = time.time() - start_time  # Total time for the workflow

    return answer, history, {
        "chunking_time": chunking_time if 'chunking_time' in locals() else 0,
        "vector_store_creation_time": vector_store_creation_time if 'vector_store_creation_time' in locals() else 0,
        "vector_store_loading_time": vector_store_loading_time if 'vector_store_loading_time' in locals() else 0,
        "retrieval_time": retrieval_time,
        "answer_time": answer_time,
        "total_time": total_time
    }

# Streamlit UI
st.set_page_config(page_title="RAG Chat with Memory + MP3", layout="centered")
st.title("🎧📄 RAG Chatbot (Text/Audio + Memory)")
st.sidebar.header("Settings")
use_existing = st.sidebar.checkbox("Use Existing Index", value=False)

# Initialize memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

transcript_text = None

if not use_existing:
    file_type, transcript_text = upload_file()
    if transcript_text:
        st.success(f"{'Audio' if file_type == 'audio' else 'Text'} file processed and indexed! You can now ask a question.")
    else:
        st.info("Please upload a .txt or .mp3 file to proceed.")

query = st.text_input("Enter your query:")
if query and st.button("Process Query"):
    response, updated_history, timings = rag_summary_workflow(
        query,
        use_existing=use_existing,
        transcript_text=transcript_text,
        history=st.session_state.conversation_history
    )
    st.session_state.conversation_history = updated_history
    st.write("**Answer:**", response)
    
    # Display time breakdown
    st.write("### Time Breakdown:")
    st.write(f"Chunking time: {timings['chunking_time']:.2f} seconds")
    st.write(f"Vector store creation time: {timings['vector_store_creation_time']:.2f} seconds")
    st.write(f"Vector store loading time: {timings['vector_store_loading_time']:.2f} seconds")
    st.write(f"Retrieval time: {timings['retrieval_time']:.2f} seconds")
    st.write(f"Answer generation time: {timings['answer_time']:.2f} seconds")
    st.write(f"Total time: {timings['total_time']:.2f} seconds")

# Display conversation memory
if st.session_state.conversation_history:
    with st.expander("🧠 Conversation History"):
        for entry in st.session_state.conversation_history:
            st.markdown(f"**User:** {entry['query']}")
            st.markdown(f"**Bot:** {entry['response']}")
            st.markdown("---")
