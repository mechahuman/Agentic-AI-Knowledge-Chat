from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from app.embeddings import LocalSentenceTransformerEmbeddings
from app.utils import get_env
import os

DB_PATH = "faiss_index"

embeddings = LocalSentenceTransformerEmbeddings()

if not os.path.exists(DB_PATH):
    raise RuntimeError("FAISS index not found. Run ingest.py first.")

db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGroq(
    api_key=get_env("GROQ_API_KEY"),
    model_name="groq/compound",
    temperature=0
)

def retrieve(query: str, k: int = 4):
    return db.similarity_search_with_score(query, k=k)

def generate_answer(query: str, docs):
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}] {doc.page_content}"
        for doc, _ in docs
    )

    prompt = f"""
You are an AI assistant answering questions strictly based on the provided context.

Rules:
- Use ONLY the information from the context below.
- You may paraphrase or summarize.
- Do NOT add external knowledge.
- If the context does not contain enough information to answer the question, respond with:
  "The information is not available in the provided document."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content
