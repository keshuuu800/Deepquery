from langchain_chroma import Chroma
from rag.embeddings import model as embedding_model

CHROMA_DATA_DIR = "chroma_db"

def store_in_vector_db(chunks: list):
    """
    Takes a list of text chunks, generates their embeddings natively,
    and stores them locally inside ChromaDB.
    """
    vector_store = Chroma(
        collection_name="wikipedia_rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DATA_DIR
    )
    vector_store.add_texts(texts=chunks)