from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from operator import itemgetter
from src.chat_history import get_session_history
from src.llm_client import call_llm_api, call_llm_api_full


#----------------------------------------------------#
# History Aware Retrieval 
#----------------------------------------------------#
rephrase_sys_msg = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question that can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", rephrase_sys_msg),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

rephrase_chain = (
    rephrase_prompt 
    | (lambda x: x.to_string()) 
    | call_llm_api_full
)

def contextualized_question(input: dict):
    if input.get("chat_history"):   # Only runs the rephrasing chain when needed
        return rephrase_chain.invoke(input)
    else:
        return input["question"]

#----------------------------------------------------#
# Query Translation: Decomposition
#----------------------------------------------------#
def format_qna(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"

def parse_subqueries(llm_output: str) -> list:
    return [line.strip() for line in llm_output.split('\n') if line.strip()]

decomp_sys_msg = (
    "You are an expert in ERCOT protocols. " # FGP: Adjust with needed area of expertise
    "Your task is to decompose the user's complex query into a few simple sub-queries that can be answered by retrieving specific context from the available knowledge base. "
    "Also make sure to expand the acronyms you find. "
    "Return the sub-queries separated by newlines."
    "\nUser's question: {question}"
    "\n-----------------\n"
    "\nOutput sub-queries (2-4): "
)
decomp_prompt = ChatPromptTemplate.from_messages([
    ("system", decomp_sys_msg),
    ("human", "{question}")
])
decomp_chain = (
    decomp_prompt
    | (lambda x: x.to_string())
    | RunnableLambda(call_llm_api_full)
    | (lambda x: parse_subqueries(x))
)

def get_subqueries(question: str) -> list:
    return decomp_chain.invoke({"question": question})

recurrent_qna_msg = (
    "You are a helpful assistant. Use the following pieces of Context to answer the user's question. "
    "If you don't know the answer based on the context, just say 'I don't have enough information to answer that request', don't try to make up an answer. "
    "Be concise, use three sentences at most. "
    "Here are some related questions and answers:"
    "\n{qna_pairs}"
    "\n-----------------\n"
    "Context: {context}"
)

qna_prompt = ChatPromptTemplate.from_messages([
    ("system", recurrent_qna_msg),
    ("human", "{question}")
])

def get_qna_chain(retriever):
    qna_chain = (
        {
            "qna_pairs": lambda x: "\n---\n".join(x["qna_pairs"]),
            "context": itemgetter("question") | retriever,
            "question": lambda x: x["question"],
        }
        | qna_prompt
        | (lambda x: x.to_string())
        | RunnableLambda(call_llm_api_full)
    )
    return qna_chain

def get_qna_pairs(subqueries: list[str], retriever) -> list:
    # subqueries = get_subqueries(question)
    qna_chain = get_qna_chain(retriever)
    qna_pairs = []
    for subquery in subqueries:
        answer = qna_chain.invoke({"question": subquery, "qna_pairs": qna_pairs})
        qna_pairs.append(format_qna(subquery, answer))
    return qna_pairs

#----------------------------------------------------#
# Final Answer Generation
#----------------------------------------------------#
def get_rag_chain(retriever):
    system_msg = recurrent_qna_msg
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    contextualize_step = RunnablePassthrough.assign(
        question=RunnableLambda(contextualized_question)
    )

    retrieval_step = (
        {
            "qna_pairs": (lambda x: x["question"]) | decomp_chain | RunnableLambda(lambda subq: get_qna_pairs(subq, retriever)),
            "context": (lambda x: x["question"]) | retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
    )

    generation_step = (
        prompt
        | (lambda x: x.to_string())
        | RunnableLambda(call_llm_api)
    )

    rag_chain = (
        contextualize_step
        | retrieval_step
        | generation_step
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history
