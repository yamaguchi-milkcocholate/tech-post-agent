from pathlib import Path
from uuid import uuid4

import faiss
import tiktoken
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..model import embedding_model
from .utils import iter_files


class VectorIndex:
    def __init__(self, saved_path: Path | None = None) -> None:
        self.dim = 32
        self.embeddings = embedding_model(dim=self.dim)
        if saved_path and saved_path.exists():
            self.vector_store = FAISS.load_local(
                folder_path=str(saved_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.index = self.vector_store.index
        else:
            self.index = faiss.IndexFlatL2(
                len(self.embeddings.embed_query("hello world"))
            )
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        self.chunk_size = 1200
        self.chunk_overlap = 150

    def build(self, root: Path) -> None:
        enc = tiktoken.get_encoding("cl100k_base")
        documents: list[Document] = []
        for p in iter_files(root):
            print(p)
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            toks = enc.encode(text)
            for i in range(0, len(toks), self.chunk_size - self.chunk_overlap):
                sub = enc.decode(toks[i : i + self.chunk_size])
                doc = Document(page_content=sub, metadata={"path": str(p)})
                documents.append(doc)

        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))

    def search(
        self, query: str, k: int = 8, filter: dict | None = None
    ) -> list[Document]:
        results = self.vector_store.similarity_search(query, k=k, filter=filter)
        return results
