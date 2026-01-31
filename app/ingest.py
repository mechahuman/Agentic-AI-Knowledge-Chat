from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings import LocalSentenceTransformerEmbeddings

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
DB_PATH = "faiss_index"

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
