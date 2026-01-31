from langgraph.graph import StateGraph
from typing import TypedDict, List, Tuple
from rag import retrieve, generate_answer


class RAGState(TypedDict):
    query: str
    docs: List[Tuple]
    answer: str
    confidence: float


def retrieve_node(state: RAGState):
    docs = retrieve(state["query"])
    return {
        "query": state["query"],
        "docs": docs,
    }


def generate_node(state: RAGState):
    docs = state.get("docs", [])

    if not docs:
        return {
            "query": state["query"],
            "docs": [],
            "answer": "The information is not available in the provided document.",
            "confidence": 0.0,
        }

    answer = generate_answer(state["query"], docs)
    scores = [score for _, score in docs]
    confidence = float(round(1 - (sum(scores) / len(scores)), 2))

    return {
        "query": state["query"],
        "docs": docs,
        "answer": answer,
        "confidence": confidence,
    }


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

rag_app = graph.compile()
