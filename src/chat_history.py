from langchain.memory import ChatMessageHistory

_store = {}

def get_session_history(session_id):
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]