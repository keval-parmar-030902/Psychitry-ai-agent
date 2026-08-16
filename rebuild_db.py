import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

print("Initializing NVIDIA Embeddings...")
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5", 
    nvidia_api_key=os.getenv("NVIDIA_API_KEY")
)

# Replace this with your actual document loading logic (e.g., PyPDFLoader)
# This is just placeholder data to initialize the DB correctly.
sample_docs = [
    Document(page_content="DSM-5 Guidelines for Major Depressive Disorder..."),
    Document(page_content="Treatment protocols for Generalized Anxiety Disorder...")
]

print("Building new ChromaDB...")
# This will create a fresh ./chroma_psych_db folder with 1024-dimension vectors
vector_db = Chroma.from_documents(
    documents=sample_docs,
    embedding=embeddings,
    collection_name="psychiatric_guidelines",
    persist_directory="./chroma_psych_db"
)

print("Database rebuilt successfully!")