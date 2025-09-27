from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

import faiss
import tiktoken
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.readers.file.base import _DefaultFileMetadataFunc

from tech_post_agent.model import chat_model, embedding_model
from tech_post_agent.schema import File
from tech_post_agent.tools.core.utils import iter_files


class RepoIndex:
    def __init__(self, chat_model_name: str = "gpt-4o-mini", dim: int = 1536) -> None:
        self.embed = embedding_model(dim=dim)
        self.chat_model = chat_model(model_name=chat_model_name)

    def _mk_index_path(self, name: str) -> Path:
        root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        store_dir = root_dir / ".cache" / "work" / "store"
        d = store_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_reader(
        self, root: Path
    ) -> tuple[
        SimpleDirectoryReader, Callable[[str | Path], SentenceSplitter | CodeSplitter]
    ]:
        name = root.name

        # コード寄りの分割器（関数/クラス境界を意識）。難しければ後で固定長に変更可。
        py_splitter = CodeSplitter(
            language="python",
            chunk_lines=300,
            chunk_lines_overlap=50,
            max_chars=8000,
        )
        txt_splitter = SentenceSplitter()

        def splitter_func(file_path: str | Path) -> SentenceSplitter | CodeSplitter:
            if str(file_path).endswith(".py"):
                return py_splitter
            else:
                return txt_splitter

        default_metadata_func = _DefaultFileMetadataFunc()
        reader = SimpleDirectoryReader(
            input_dir=root,
            exclude_hidden=False,
            recursive=True,
            required_exts=[
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".cc",
                ".c",
                ".h",
                ".md",
                ".txt",
                ".yml",
                ".yaml",
                ".toml",
                ".json",
            ],
            filename_as_id=True,
            file_extractor={},  # 既定のテキスト抽出を利用
            file_metadata=lambda x: {
                "repo_name": name,
                "relative_path": str(Path(x).relative_to(root)),
            }
            | default_metadata_func(x),
        )
        return reader, splitter_func

    @lru_cache(maxsize=16)
    def load_or_create_index(self, root: Path) -> VectorStoreIndex:
        name = root.name

        store_dir = self._mk_index_path(name=name)

        if store_dir.exists() and any(store_dir.iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(store_dir))
            return load_index_from_storage(
                storage_context=storage_context, embed_model=self.embed
            )
        else:
            reader, splitter_func = self._build_reader(root=root)
            docs = reader.load_data()

            nodes = []
            for doc in docs:
                splitter = splitter_func(doc.metadata["file_name"])
                nodes.extend(splitter.get_nodes_from_documents([doc]))
            index = VectorStoreIndex(nodes, embed_model=self.embed)
            index.storage_context.persist(persist_dir=str(store_dir))
            return index

    def search(
        self,
        repo_root: str,
        query: str,
        top_k: int = 8,
    ) -> list[File]:
        idx = self.load_or_create_index(repo_root)
        hits = idx.as_retriever(similarity_top_k=top_k).retrieve(query)
        out = []
        for h in hits:
            meta = h.node.metadata or {}
            file = File(
                meta={
                    "repo_name": meta.get("repo_name"),
                    "name": meta.get("file_name"),
                    "relative_path": meta.get("relative_path"),
                    "absolute_path": meta.get("file_path"),
                    "type": meta.get("file_type"),
                },
                content=h.node.get_content(),
            )
            out.append(file)
        return out

    def answer(self, repo_root: str, question: str, top_k: int = 8) -> str:
        idx = self.load_or_create_index(repo_root)
        qe = idx.as_query_engine(similarity_top_k=top_k, llm=self.chat_model)
        resp = qe.query(question)
        sources = []
        for s in getattr(resp, "source_nodes", []) or []:
            m = s.node.metadata or {}
            p = m.get("file_name") or m.get("filename")
            if p and p not in sources:
                sources.append(p)
        return f"{str(resp)}\n\n[SOURCES]\n" + "\n".join(f"- {p}" for p in sources)


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
