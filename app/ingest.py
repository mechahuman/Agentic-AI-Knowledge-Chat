from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.embeddings import LocalSentenceTransformerEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "Ebook-Agentic-AI.pdf")
DB_PATH = os.path.join(BASE_DIR, "faiss_index")

def ingest():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = LocalSentenceTransformerEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    print(f"Ingested {len(chunks)} chunks")

if __name__ == "__main__":
    ingest()
