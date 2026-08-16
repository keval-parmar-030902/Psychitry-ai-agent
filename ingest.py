import os
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DOCUMENTS_DIR = "./clinical_documents"
DB_DIR = "./chroma_psych_db"
COLLECTION_NAME = "psychiatric_guidelines"

def ingest_documents():
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        logger.info(f"Created directory {DOCUMENTS_DIR}. Please add your PDFs here.")
        return

    logger.info("Loading PDFs from directory...")
    # Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
    documents = loader.load()

    if not documents:
        logger.warning(f"No documents found in {DOCUMENTS_DIR}.")
        return

    logger.info(f"Loaded {len(documents)} document pages. Splitting text...")
    
    # Split documents into smaller chunks for the LLM to process easily
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, # Overlap prevents cutting concepts in half
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Split documents into {len(chunks)} chunks. Generating embeddings...")

    # Initialize the same embedding model used in rag_engine.py
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store the chunks in ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )
    
    logger.info(f"Successfully ingested and saved {len(chunks)} chunks to {DB_DIR}.")

if __name__ == "__main__":
    ingest_documents()