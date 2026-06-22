import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(raw_text: str) -> list:
    """Splits a large string of text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)