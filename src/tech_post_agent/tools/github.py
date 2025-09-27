from pathlib import Path

from langchain_core.tools import tool
from requests.exceptions import HTTPError

from tech_post_agent.schema import Repo
from tech_post_agent.tools.core import GitHubIngestor


@tool
def download_github_repo(repo_url: str) -> Repo:
    """Download a GitHub repository."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    work_dir = root_dir / ".cache" / "work" / "github"

    ingestor = GitHubIngestor(work_dir=work_dir)
    try:
        path = ingestor.from_github(repo_url=repo_url, branch="main")
    except HTTPError:
        path = ingestor.from_github(repo_url=repo_url, branch="master")

    print("[GitHubIngestor] Downloaded repository to", path)

    return Repo(name=path.name, url=repo_url, local_path=str(path))
