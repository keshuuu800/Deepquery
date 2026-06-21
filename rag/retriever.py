import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "wiki_articles"
)

def retrieve(
    question,
    article
):

    results = collection.query(
        query_texts=[question],
        n_results=5,
        where={
            "article": article
        }
    )

    return results["documents"][0]