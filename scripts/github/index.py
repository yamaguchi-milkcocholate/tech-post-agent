from pathlib import Path

from tech_post_agent.tools.core.index import RepoIndex


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    repo_dir = root_dir / ".cache/work/github/ingest/openpoke-main"

    repo_index = RepoIndex()

    search_resp = repo_index.search(
        repo_root=repo_dir, query="OpenPokeについて教えてください", top_k=1
    )
    print(search_resp)

    search_resp = repo_index.answer(
        repo_root=repo_dir, question="OpenPokeについて教えてください", top_k=1
    )
    print(search_resp)


if __name__ == "__main__":
    main()
