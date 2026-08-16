import os
import tempfile
import streamlit as st
from unity.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Configuration ---
DB_DIR = "./chroma_psych_db"
COLLECTION_NAME = "psychiatric_guidelines"

st.set_page_config(page_title="Admin: Knowledge Base Updater", page_icon="🗂️")

st.title("🗂️ Admin: Update Clinical Knowledge Base")
st.markdown("""
Use this portal to upload new clinical guidelines, research papers, or disease criteria (PDF format). 
The AI agent will immediately have access to this new knowledge for patient triage and treatment drafting.
""")

# Initialize embedding model
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

uploaded_files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Ingest Documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    else:
        with st.spinner("Processing and embedding documents..."):
            all_chunks = []
            
            # 1. Process each uploaded file
            for uploaded_file in uploaded_files:
                # Save uploaded file to a temporary file to use with PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # 2. Load the PDF
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    # 3. Split the text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    chunks = text_splitter.split_documents(documents)
                    all_chunks.extend(chunks)
                    
                    st.success(f"Successfully processed: {uploaded_file.name} ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up temp file
                    os.remove(tmp_file_path)
            
            # 4. Save to ChromaDB Vector Store
            if all_chunks:
                try:
                    embeddings = get_embeddings()
                    Chroma.from_documents(
                        documents=all_chunks,
                        embedding=embeddings,
                        collection_name=COLLECTION_NAME,
                        persist_directory=DB_DIR
                    )
                    st.success(f"✅ Successfully added {len(all_chunks)} total chunks to the Vector Database!")
                    st.info("The patient intake agent now has access to this new information.")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")