from pathlib import Path
from typing import Callable

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter

from tech_post_agent.model import embedding_model


def _mk_index_path(name: str) -> Path:
    root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    store_dir = root_dir / ".cache" / "work" / "store"
    d = store_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_reader(
    root: Path,
) -> tuple[
    SimpleDirectoryReader, Callable[[str | Path], SentenceSplitter | CodeSplitter]
]:
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
    )
    return reader, splitter_func


def load_or_create_index(root: Path) -> VectorStoreIndex:
    name = root.name
    store_dir = _mk_index_path(name=name)
    embed = embedding_model(dim=1536)

    if store_dir.exists() and any(store_dir.iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=str(store_dir))
        return load_index_from_storage(
            storage_context=storage_context, embed_model=embed
        )
    else:
        reader, splitter_func = _build_reader(root)
        docs = reader.load_data()

        nodes = []
        for doc in docs:
            splitter = splitter_func(doc.metadata["file_name"])
            nodes.extend(splitter.get_nodes_from_documents([doc]))
        index = VectorStoreIndex(nodes, embed_model=embed)
        index.storage_context.persist(persist_dir=str(store_dir))
        return index


# @tool
# def index_repo(repo_root: str) -> str:
#     """
#     指定ディレクトリをインデックス化し、永続化する。
#     既存があれば再利用。戻り値は repo_id（例: owner__repo）
#     """
#     idx = _load_or_create_index(repo_root)
#     return _repo_id_from_path(repo_root)


# @tool
# def search_repo(repo_root: str, query: str, top_k: int = 8) -> List[dict]:
#     """
#     インデックスからクエリ検索して上位を返す。各ヒットのpath/score/previewを含む。
#     """
#     idx = _load_or_create_index(repo_root)
#     engine = idx.as_retriever(similarity_top_k=max(1, top_k))
#     hits = engine.retrieve(query)
#     out = []
#     for h in hits:
#         meta = h.node.metadata or {}
#         out.append(
#             {
#                 "path": meta.get("file_name") or meta.get("filename") or "unknown",
#                 "score": float(h.score or 0.0),
#                 "preview": (h.node.get_content() or "")[:500],
#             }
#         )
#     return out


# @tool
# def answer_with_context(
#     repo_root: str, question: str, top_k: int = 8, system_hint: str | None = None
# ) -> str:
#     """
#     検索→関連ノードをコンテキストに回答生成（簡易版）。
#     出典（path）の列挙を含める。
#     """
#     idx = _load_or_create_index(repo_root)
#     query_engine = idx.as_query_engine(
#         similarity_top_k=max(1, top_k),
#         text_qa_template=None,  # 既定テンプレ
#     )
#     if system_hint:
#         # ChatEngineに切替するほどでなければsystemを混ぜる簡易ハック
#         pass
#     resp = query_engine.query(question)

#     # 出典ファイルを併記（重複排除）
#     sources = []
#     try:
#         for s in resp.source_nodes:
#             meta = s.node.metadata or {}
#             path = meta.get("file_name") or meta.get("filename")
#             if path and path not in sources:
#                 sources.append(path)
#     except Exception:
#         pass

#     return f"{str(resp)}\n\n[SOURCES]\n" + "\n".join(f"- {p}" for p in sources)
