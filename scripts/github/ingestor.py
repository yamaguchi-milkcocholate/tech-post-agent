from pathlib import Path

from tech_post_agent.tools.core.ingestor import GitHubIngestor
from tech_post_agent.tools.index import VectorIndex

work_dir = Path(__file__).resolve().parent.parent.parent / ".cache" / "work" / "github"


def main():
    ingest_work_dir = work_dir / "ingest"
    ingest_work_dir.mkdir(parents=True, exist_ok=True)

    ingestor = GitHubIngestor(work_dir=ingest_work_dir)
    path = ingestor.from_github(
        repo_url="https://github.com/shlokkhemani/openpoke", branch="main"
    )

    index_path = work_dir / "faiss_index"
    if index_path.exists():
        vector_index = VectorIndex(saved_path=index_path)
    else:
        vector_index = VectorIndex()
        vector_index.build(root=path)
        vector_index.save(path=index_path)

    docs = vector_index.search(
        "README usage install quickstart",
        k=3,
    )
    for i, doc in enumerate(docs):
        print("-----")
        print(f"Document {i + 1}:")
        print(doc.metadata)
        print(doc.page_content)
        print("-----")


if __name__ == "__main__":
    main()
