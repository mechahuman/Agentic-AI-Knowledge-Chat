from fastapi import FastAPI
from pydantic import BaseModel
from graph import rag_app

app = FastAPI(title = "Agentic AI")


class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: QueryRequest):
    result = rag_app.invoke({"query": req.question})

    return {
        "question": req.question,
        "answer": result.get("answer", "An error occurred while generating the answer."),
        "confidence": result.get("confidence", 0.0),
        "contexts": [
            {
                "text": doc.page_content,
                "page": doc.metadata.get("page"),
                "score": float(score)
            }
            for doc, score in result.get("docs", [])
        ]
    }
