import os
import hashlib
from typing import List, Optional, Dict

from langchain_core.documents import Document

from src.vector_store import get_vectorstore
from src.add_documents.text_extractor import ProductionTextExtractor
from src.add_documents.chunker import SGIAChunker
from pathlib import Path

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def upsert_documents_to_chroma(
    paths: List[str],
    original_names: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None
) -> Dict[str, int]:
    """
    Index docs and return chunks count per file path.
    """
    vectorstore = get_vectorstore()

    stats: Dict[str, int] = {}
    all_chunks: List[Document] = []
    all_ids: List[str] = []

    if original_names is None:
        original_names = [os.path.basename(p) for p in paths]
    if len(original_names) != len(paths):
        raise ValueError("original_names must have the same length as paths")

    for p, original_name in zip(paths, original_names):
        print(f"Processing temp path: {p}")
        print(f"Original filename: {original_name}")
        chunks = chunk_documents(p, original_name)

        if not chunks:
            stats[p] = 0
            continue

        # opcional: añade metadata extra
        if extra_metadata:
            for c in chunks:
                c.metadata.update(extra_metadata)

        stats[p] = len(chunks)

        for i, c in enumerate(chunks):
            fid = c.metadata.get("source_file_id", "nofile")
            section = c.metadata.get("section", "nosection")
            chunk_index = c.metadata.get("chunk_index", i)

            chunk_id = c.metadata.get("chunk_id")
            all_ids.append(chunk_id if chunk_id else f"{fid}::{section}::{chunk_index}")
            all_chunks.append(c)

    if not all_chunks:
        raise RuntimeError("No chunks were created (empty extraction or chunking). Nothing to index.")

    vectorstore.add_documents(all_chunks, ids=all_ids)
    try:
        vectorstore.persist()
    except Exception:
        pass

    return stats


def delete_document_from_chroma(source_file_id: str) -> int:
    """
    Deletes all chunks belonging to a document (identified by source_file_id)
    from the persistent ChromaDB.
    Returns the number of deleted records if available (otherwise 0).
    """
    vectorstore = get_vectorstore()

    # LangChain's Chroma wrapper doesn't always expose metadata deletes nicely,
    # but the underlying Chroma collection does.
    try:
        # Preferred: delete by metadata filter
        # This works if your Chroma version supports `where`
        result = vectorstore._collection.delete(where={"source_file_id": source_file_id})
        # result may be None depending on version
        return 0 if result is None else 0
    except Exception:
        # Fallback: try deleting by id prefix if where isn't supported.
        # This fallback is limited because we don't know how many chunks exist,
        # so we can't enumerate ids reliably without re-chunking.
        # Better to keep the "where" method above working.
        raise RuntimeError(
            "Could not delete by metadata filter. "
            "Your Chroma version may not support where-delete via _collection."
        )


def chunk_documents(path: str, original_name: str) -> List[Document]:
    """
    Extract text + chunk it using the project's official SGIA chunker.
    Returns LangChain Documents with rich metadata.
    """
    extractor = ProductionTextExtractor()
    chunker = SGIAChunker()

    pdf_path = Path(path)
    print(f"Processing file: {pdf_path}")
    result = extractor.extract_document(pdf_path) 
    text = result.full_text                         # extrae el texto real del DocumentResult

    print("DEBUG result fields:", dir(result))
    print("DEBUG full_text len:", len(getattr(result, "full_text", "") or ""))
    print("DEBUG text len:", len(getattr(result, "text", "") or ""))
    print("DEBUG pages:", len(getattr(result, "pages", []) or []))

    print(f" - Extracted {len(text)} characters from {path}")
    chunks = chunker.chunk_document(text=text, filename=original_name)
    print(f" - Created {len(chunks)} chunks from {path}")

    # Convertir chunks (dicts) -> LangChain Documents
    docs: List[Document] = []
    file_id = _file_sha1(path)

    for ch in chunks:
        # ch es un objeto Chunk (dataclass), no dict
        meta = getattr(ch, "metadata", {}) or {}

        meta.update({
            "source_path": str(pdf_path),
            "source_file_id": file_id,
            "source_name": original_name,
            "chunk_id": getattr(ch, "chunk_id", None),
            "section": meta.get("section", "nosection"),
            "chunk_index": meta.get("chunk_index", None),
        })

        docs.append(Document(page_content=ch.text, metadata=meta))

    return docs
