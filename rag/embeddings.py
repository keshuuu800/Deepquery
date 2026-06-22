from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def create_embeddings(chunks: list) -> list:
    """Generate embeddings for a list of text chunks."""
    return model.embed_documents(chunks)
