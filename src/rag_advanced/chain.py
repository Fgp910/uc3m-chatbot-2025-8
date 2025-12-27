# Chain builders for RAG pipeline

from typing import Dict, Generator
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableGenerator
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import get_session_history
from src.llm_client import call_llm_api_full
from .utils import (
    get_logger, detect_language, format_sources, RAGMode, config
)
from .prompts import REPHRASE_PROMPT
from .components import (
    is_domain_relevant, contextualize_question, 
    generate_flash_response, generate_thinking_response
)

# --- Chain Builders ---

def get_flash_chain(retriever, with_history: bool = True, with_summary: bool = False):
    """Build Flash mode RAG chain (fast, 2-4 LLM calls with decomposition).
    
    Args:
        retriever: Document retriever
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
    """
    logger = get_logger()
    logger.info("Building FLASH mode chain")
    
    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter."""
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)
        
        # Domain pre-filter: skip retrieval for out-of-scope questions
        if not is_domain_relevant(question, history):
            msg = ("Esta pregunta no está relacionada con acuerdos de interconexión ERCOT. "
                   "Solo puedo responder preguntas sobre proyectos de energía, redes eléctricas y ERCOT."
                   if lang == 'spanish'
                   else "This question is not related to ERCOT interconnection agreements. "
                   "I can only answer questions about energy projects, power grids, and ERCOT.")
            yield msg
            return
        
        # Retrieve documents directly
        docs = retriever.invoke(question)
        retrieval = format_sources(docs, max_sources=config.FLASH_MAX_SOURCES)
        
        # Generate response
        for chunk in generate_flash_response({
            "question": question,
            "retrieval": retrieval,
            "chat_history": history,
            "with_summary": with_summary
        }):
            yield chunk
    
    rag_chain_core = (
        RunnablePassthrough.assign(
            question=RunnableLambda(contextualize_question)
        )
        | RunnableLambda(flash_with_domain_filter)
    )
    
    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_thinking_chain(retriever, with_history: bool = True, with_summary: bool = False):
    """Build Thinking mode RAG chain (deep verification, 5-10 LLM calls).
    
    Args:
        retriever: Document retriever
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
    """
    logger = get_logger()
    logger.info("Building THINKING mode chain")
    
    def thinking_generator(input_iter):
        """Generator function for RunnableGenerator - yields chunks from thinking response."""
        for input_dict in input_iter:
            question = input_dict.get("question", "")
            lang = detect_language(question)
            
            # Domain guardrail FIRST - before any LLM calls (with chat context)
            history = input_dict.get("chat_history", [])
            if not is_domain_relevant(question, history):
                msg = ("Esta pregunta no está relacionada con acuerdos de interconexión ERCOT. "
                       "Solo puedo responder preguntas sobre proyectos de energía, redes eléctricas y ERCOT."
                       if lang == 'spanish'
                       else "This question is not related to ERCOT interconnection agreements. "
                       "I can only answer questions about energy projects, power grids, and ERCOT.")
                yield msg
                return
            
            # Contextualize question (only if relevant)
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}
            
            # Pass summary option to thinking response
            input_dict = {
                **input_dict,
                "with_summary": with_summary
            }
            
            # Generate thinking response
            for chunk in generate_thinking_response(input_dict, retriever):
                yield chunk
    
    rag_chain_core = RunnableGenerator(thinking_generator)
    
    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_rag_chain(retriever, mode: RAGMode = RAGMode.FLASH, with_history: bool = True, with_summary: bool = False):
    """Get RAG chain with specified mode.
    
    Args:
        retriever: The document retriever
        mode: RAGMode.FLASH or RAGMode.THINKING
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary to responses
    """
    if mode == RAGMode.FLASH:
        return get_flash_chain(retriever, with_history, with_summary)
    else:
        return get_thinking_chain(retriever, with_history, with_summary)
