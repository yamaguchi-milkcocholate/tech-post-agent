from pathlib import Path

from tech_post_agent.tools.core.ingestor import GitHubIngestor

work_dir = Path(__file__).resolve().parent.parent.parent / ".cache" / "work" / "github"


def main():
    ingest_work_dir = work_dir / "ingest"
    ingest_work_dir.mkdir(parents=True, exist_ok=True)

    ingestor = GitHubIngestor(work_dir=ingest_work_dir)
    path = ingestor.from_github(
        repo_url="https://github.com/shlokkhemani/openpoke", branch="main"
    )
    print(f"ingested to: {path}")


if __name__ == "__main__":
    main()
