# models/chroma_loader.py
import chromadb

def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_20newsgroups")
    return client.get_collection("twenty_newsgroups")
