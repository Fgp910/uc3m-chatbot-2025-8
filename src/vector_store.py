# Mock ChromaDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document


# 2. Create some dummy documents
mock_documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "document1"}),
    Document(page_content="Artificial intelligence is rapidly changing the world.", metadata={"source": "document2"}),
    Document(page_content="Machine learning is a subset of AI that focuses on algorithms.", metadata={"source": "document3"}),
    Document(page_content="Natural Language Processing (NLP) deals with human language.", metadata={"source": "document4"}),
    Document(page_content="RAG systems combine retrieval and generation for better answers.", metadata={"source": "document5"}),
    Document(page_content="ChromaDB is a popular open-source vector database.", metadata={"source": "document6"})
]

def get_vectorstore():
    # 1. Define the embedding function (must be the same as used in retrieve_relevant_documents)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Define the directory for persistence
    persist_directory = "./chroma_db"

    # 4. Initialize and populate ChromaDB
    print(f"Creating mock ChromaDB in '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=mock_documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    # Persist the collection to disk
    vectorstore.persist()
    print("Mock ChromaDB created and populated successfully!")
    print(f"Number of documents in ChromaDB: {vectorstore._collection.count()}")

    return vectorstore

def get_retriever(k_docs: int = 3):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})
    return retriever
