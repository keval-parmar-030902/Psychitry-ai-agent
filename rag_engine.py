from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

# Initialize local embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Setup Vector DB (You would pre-populate this with clinical documents in a real scenario)
vector_db = Chroma(
    collection_name="psychiatric_guidelines",
    embedding_function=embeddings,
    persist_directory="./chroma_psych_db"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

@tool
def search_clinical_guidelines(query: str) -> str:
    """Useful to look up psychiatric clinical practice guidelines, DSM-5 criteria, and evidence-based treatment regimens."""
    docs = retriever.invoke(query)
    if not docs:
        return "No specific guidelines found for this query."
    return "\n\n".join([d.page_content for d in docs])