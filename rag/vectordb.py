import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    "wikipedia"
)

import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    "wiki_articles"
)

def store_chunks(
    title,
    chunks
):

    ids = [
        f"{title}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "article": title
        }
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )