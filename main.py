from src.rag import get_rag_chain, get_rag_chain_with_summary
from src.vector_store import get_retriever

K_DOCS = 10

print("Loading retriever...")
retriever = get_retriever(k_docs=K_DOCS)

print("Building RAG chain...")
chain = get_rag_chain(retriever)

# Test 1: Basic query with source citations
print("\n" + "="*60)
print("TEST 1: Security requirements (English)")
print("="*60)
question = "What are the security deposit requirements for interconnection?"
print("Question: " + question)
for chunk in chain.stream(
    {"question": question},
    config={"configurable": {"session_id": "test1"}}
):
    print(chunk, end="", flush=True)

# Test 2: Follow-up (tests conversation history)
print("\n\n" + "="*60)
print("TEST 2: Follow-up question")
print("="*60)
question = "What happens if the developer fails to meet those requirements?"
print("Question: " + question)
for chunk in chain.stream(
    {"question": question},
    config={"configurable": {"session_id": "test1"}}
):
    print(chunk, end="", flush=True)

# Test 3: Spanish query (tests language detection)
print("\n\n" + "="*60)
print("TEST 3: Spanish query")
print("="*60)
question = "Cuales son los requisitos de seguridad para proyectos solares?"
print("Question: " + question)
for chunk in chain.stream(
    {"question": question},
    config={"configurable": {"session_id": "test2"}}
):
    print(chunk, end="", flush=True)

# Test 4: Query with no relevant info (tests "no information" handling)
print("\n\n" + "="*60)
print("TEST 4: Out-of-scope question")
print("="*60)
question = "What is the capital of France?"
print("Question: " + question)
for chunk in chain.stream(
    {"question": question},
    config={"configurable": {"session_id": "test3"}}
):
    print(chunk, end="", flush=True)

# Test 5: With auto-summarization (extension for 10/10)
print("\n\n" + "="*60)
print("TEST 5: With auto-summarization")
print("="*60)
chain_summary = get_rag_chain_with_summary(retriever)
question = "What are the milestone requirements in Article 5?"
print("Question: " + question)
for chunk in chain_summary.stream(
    {"question": question},
    config={"configurable": {"session_id": "test4"}}
):
    print(chunk, end="", flush=True)

print("\n\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
