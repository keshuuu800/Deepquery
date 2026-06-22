from langchain_chroma import Chroma
from rag.embeddings import model as embedding_model
from rag.chunker import chunk_text
CHROMA_DATA_DIR = "chroma_db"
def retrieve_relevant_chunks(query: str, k: int = 10) -> list:
    """
    Searches ChromaDB for the top 'k' most relevant text chunks.
    """
    vector_store = Chroma(
        collection_name="wikipedia_rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DATA_DIR
    )
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]