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
    "uv.lock",
}
FILE_IGNORES = {"package-lock.json", ".DS_Store"}


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


def is_code(path: Path) -> bool:
    return path.suffix.lower() in CODE_EXTS


def is_doc(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTS
