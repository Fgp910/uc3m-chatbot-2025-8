from src.rag import get_rag_chain
from src.vector_store import get_retriever

K_DOCS = 5  # Number of documents to retrieve

retriever = get_retriever(k_docs=K_DOCS)
chain_with_history = get_rag_chain(retriever)
# Turn 1
print("--- Turn 1 ---")
for chunk in chain_with_history.stream(
    {"question": "What is Artificial Intelligence?"},
    config={"configurable": {"session_id": "FGP"}}
):
    print(chunk, end="", flush=True)

# Turn 2 (Follow up)
print("\n\n--- Turn 2 ---")
for chunk in chain_with_history.stream(
    {"question": "How does machine learning relate to it?"}, # Asking about previous context
    config={"configurable": {"session_id": "FGP"}}
):
    print(chunk, end="", flush=True)