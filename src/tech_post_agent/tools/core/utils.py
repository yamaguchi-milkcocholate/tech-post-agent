import shutil
from pathlib import Path
from typing import Iterable

CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".rb",
    ".php",
}
DOC_EXTS = {".md", ".rst", ".txt"}
BIN_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".lock"}

DIR_IGNORES = {
    "node_modules",
    ".git",
    ".venv",
    "dist",
    "build",
    "site-packages",
}
FILE_IGNORES = {"package-lock.json", ".DS_Store"}


def rm_excluded_files(root: Path) -> Iterable[Path]:
    rm_paths = []
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in DIR_IGNORES:
                rm_paths.append(p)
        elif p.is_file() and (p.name in FILE_IGNORES):
            rm_paths.append(p)
        else:
            if p.suffix.lower() in BIN_EXTS:
                rm_paths.append(p)

    for p in rm_paths:
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p, ignore_errors=True)
        print(f"rm {p}")


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in DIR_IGNORES:
                continue
        elif p.is_file() and (p.name in FILE_IGNORES):
            continue
        else:
            if p.suffix.lower() in BIN_EXTS:
                continue
            yield p
