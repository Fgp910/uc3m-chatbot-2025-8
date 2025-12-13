"""
RAG Pipeline - Production Grade

Requirements:
1. Source citations in every response
2. "No information" when docs don't answer
3. Response in same language as question
4. Auto-summarization extension
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from src.chat_history import get_session_history
from src.llm_client import call_llm_api, call_llm_api_full


def detect_language(text: str) -> str:
    text_lower = text.lower()
    spanish_words = ['qué', 'cómo', 'cuál', 'dónde', 'por qué', 'el', 'la', 'los', 
                     'proyecto', 'seguridad', 'requisitos', 'cuánto', 'para']
    spanish_count = sum(1 for w in spanish_words if w in text_lower)
    return 'spanish' if spanish_count >= 2 else 'english'


def format_sources(docs) -> tuple:
    """Returns (formatted_context, source_list)"""
    if not docs:
        return "", []
    
    formatted_parts = []
    source_list = []
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        project = meta.get('project_name', 'Unknown')
        inr = meta.get('inr', 'N/A')
        section = meta.get('section', 'N/A')
        
        formatted_parts.append(f"[Source {i}: {project} ({inr}) - {section}]\n{doc.page_content}\n")
        source_list.append({
            'ref': i,
            'project_name': project,
            'inr': inr,
            'section': section,
            'zone': meta.get('zone'),
            'fuel_type': meta.get('fuel_type')
        })
    
    return "\n".join(formatted_parts), source_list


def format_citations(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for s in sources[:5]:
        lines.append(f"  [{s['ref']}] {s['project_name']} ({s['inr']}) - {s['section']}")
    return "\n".join(lines)


# History-aware rephrasing
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question to be standalone given the chat history. Return only the reformulated question."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

rephrase_chain = rephrase_prompt | (lambda x: x.to_string()) | call_llm_api_full

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return rephrase_chain.invoke(input)
    return input["question"]


# Query decomposition
decomp_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decompose this ERCOT interconnection question into 2-3 simple sub-queries. Return one per line.\n\nQuestion: {question}"),
    ("human", "{question}")
])

decomp_chain = (
    decomp_prompt
    | (lambda x: x.to_string())
    | RunnableLambda(call_llm_api_full)
    | (lambda x: [line.strip() for line in x.split('\n') if line.strip()])
)


# System prompts
SYSTEM_EN = """Answer questions about ERCOT interconnection agreements using ONLY the provided context.

RULES:
- If context doesn't contain the answer, say: "I don't have information about that in the available documents."
- Cite sources using [Source N] format
- Be concise

Context:
{context}"""

SYSTEM_ES = """Responde preguntas sobre acuerdos de interconexión ERCOT usando SOLO el contexto proporcionado.

REGLAS:
- Si el contexto no contiene la respuesta, di: "No tengo información sobre eso en los documentos disponibles."
- Cita fuentes usando formato [Fuente N]
- Sé conciso

Contexto:
{context}"""


def get_rag_chain(retriever, with_summary: bool = False):
    """
    Build RAG chain with source citations and language detection.
    
    Args:
        retriever: LangChain retriever
        with_summary: Enable auto-summarization extension
    """
    
    class RAGChain:
        def __init__(self, retriever, with_summary):
            self.retriever = retriever
            self.with_summary = with_summary
        
        def stream(self, inputs: dict, config: dict = None):
            session_id = config.get('configurable', {}).get('session_id', 'default') if config else 'default'
            history = get_session_history(session_id)
            
            # Contextualize question
            question = inputs['question']
            if history.messages:
                question = contextualized_question({
                    'question': question,
                    'chat_history': history.messages
                })
            
            # Detect language
            lang = detect_language(question)
            
            # Retrieve documents (use invoke instead of deprecated get_relevant_documents)
            docs = self.retriever.invoke(question)
            context, sources = format_sources(docs)
            
            # Handle no documents
            if not docs:
                no_info = "No tengo información sobre eso en los documentos disponibles." if lang == 'spanish' else "I don't have information about that in the available documents."
                history.add_user_message(inputs['question'])
                history.add_ai_message(no_info)
                yield no_info
                return
            
            # Build prompt
            system = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("placeholder", "{chat_history}"),
                ("human", "{question}")
            ])
            
            prompt_text = prompt.invoke({
                'context': context,
                'chat_history': history.messages,
                'question': question
            }).to_string()
            
            # Generate response
            response_parts = []
            for token in call_llm_api(prompt_text):
                response_parts.append(token)
                yield token
            
            # Add citations
            citations = format_citations(sources)
            yield citations
            response_parts.append(citations)
            
            # Auto-summarization extension
            if self.with_summary and docs:
                yield "\n\n--- Summary ---\n"
                summary_prompt = f"Summarize key points from these ERCOT documents in 2-3 sentences:\n{context[:3000]}"
                summary = call_llm_api_full(summary_prompt)
                yield summary
                response_parts.append(f"\n\n--- Summary ---\n{summary}")
            
            # Update history
            history.add_user_message(inputs['question'])
            history.add_ai_message(''.join(response_parts))
        
        def invoke(self, inputs: dict, config: dict = None):
            return ''.join(self.stream(inputs, config))
    
    return RAGChain(retriever, with_summary)


def get_rag_chain_with_summary(retriever):
    """RAG chain with auto-summarization enabled."""
    return get_rag_chain(retriever, with_summary=True)
