import streamlit as st

from graph import rag_app

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    layout="wide"
)

st.title("Agentic AI RAG Chatbot")
st.caption("Answers strictly based on the Agentic AI eBook")

query = st.text_input(
    "Ask a question from the document",
    placeholder="e.g. What is Agentic AI?"
)

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        try:
            result = rag_app.invoke({"query": query})

            st.subheader("Answer")
            st.write(result.get("answer", "No answer generated."))

            confidence = float(result.get("confidence", 0.0))
            st.subheader("Confidence")
            st.progress(min(confidence, 1.0))
            st.caption(f"Confidence Score: {round(confidence, 2)}")

            st.subheader("Retrieved Context")
            docs = result.get("docs", [])

            if not docs:
                st.info("No relevant context chunks retrieved.")
            else:
                for idx, (doc, score) in enumerate(docs, 1):
                    with st.expander(
                        f"Chunk {idx} | Page {doc.metadata.get('page')} | Score {round(float(score), 3)}"
                    ):
                        st.write(doc.page_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
