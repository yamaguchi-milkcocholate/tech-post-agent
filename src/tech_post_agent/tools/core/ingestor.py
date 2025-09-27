import zipfile
from pathlib import Path

import requests


class GitHubIngestor:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    def from_github(self, repo_url: str, branch: str | None = None) -> Path:
        # https://github.com/owner/name(.git) → zipダウンロードに変換
        u = repo_url.rstrip("/").removesuffix(".git")
        owner, name = u.split("github.com/")[-1].split("/")[:2]
        branch = branch or "HEAD"
        zip_url = f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/{branch}"
        zpath = self.workdir / f"{owner}-{name}-{branch}.zip"

        r = requests.get(zip_url, stream=True)
        r.raise_for_status()

        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(self.workdir)

        unpacked = next(self.workdir.glob(f"{name}-*"))
        return unpacked
