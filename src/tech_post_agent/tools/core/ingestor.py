import zipfile
from pathlib import Path

import requests


class GitHubIngestor:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def from_github(self, repo_url: str, branch: str | None = None) -> Path:
        # https://github.com/owner/name(.git) → zipダウンロードに変換
        u = repo_url.rstrip("/").removesuffix(".git")
        owner, name = u.split("github.com/")[-1].split("/")[:2]
        branch = branch or "main"
        zip_url = f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/{branch}"
        zpath = self.work_dir / f"{owner}-{name}-{branch}.zip"

        r = requests.get(zip_url, stream=True)
        r.raise_for_status()

        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(self.work_dir)

        unpacked = next(self.work_dir.glob(f"{name}-*"))
        return unpacked
