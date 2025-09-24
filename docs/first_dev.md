# 目的

GitHub の任意のリポジトリを読み込み、

1. 構成と依存関係の把握、2) 使い方の要約、3) 主要 API の抽出、4) サンプルコードの自動生成、5) README/ドキュメント雛形の生成
   までを **1 コマンドで実行** する最小構成の AI エージェントです。

---

# アーキテクチャ（MVP）

```
repo-insight-agent/
├── pyproject.toml                # uv / hatch どちらでもOK（ここではuv想定）
├── .env.example                  # OPENAI_API_KEY など
├── src/
│   └── agent/
│       ├── main.py              # CLIエントリ（workflowオーケストレーション）
│       ├── ingest.py            # 取得（clone / zip / GitHub API）とファイル選別
│       ├── index.py             # チャンク化・埋め込み・ベクトルDB (FAISS / Chroma)
│       ├── analyze.py           # 言語判定・依存抽出・エントリポイント探索
│       ├── generate.py          # 要約/サンプル/README生成（プロンプト＆ガードレール）
│       ├── prompts.py           # システム/テンプレ/スケルトン
│       └── utils.py             # 共通ユーティリティ
├── outputs/
│   ├── repo_overview.md         # リポジトリ概要
│   ├── samples/                 # 生成されたサンプルコード
│   └── docs/README.draft.md     # 生成READMEドラフト
└── tests/
    └── smoke_test.py
```

---

# 依存関係（例: uv）

```toml
# pyproject.toml
[project]
name = "repo-insight-agent"
version = "0.1.0"
description = "Read a GitHub repo, understand it, and auto-generate samples & docs."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "faiss-cpu>=1.8.0",          # ベクトルDB（ローカル）
  "tiktoken>=0.7.0",           # トークン化
  "unstructured[all]>=0.15.0", # マルチフォーマット読み取り
  "python-dotenv>=1.0.1",
  "requests>=2.32.3",
  "pydantic>=2.9.2",
  "typer>=0.12.4",
  "rich>=13.8.0",
  "pygments>=2.18.0",
  "tree_sitter>=0.21.3",       # 構文解析（任意）
  "openai>=1.51.0",            # モデル呼び出し（プロバイダ差し替え可）
]

[tool.uv]
index-strategy = "unsafe-best-match"

[project.optional-dependencies]
chroma = ["chromadb>=0.5.5"]
```

`.env.example`

```
OPENAI_API_KEY=sk-xxxx
GITHUB_TOKEN=ghp_xxxx               # rate-limit回避やプライベート用
MODEL_NAME=gpt-4.1-mini             # 任意（Claude/Vertex等に差し替え可能）
EMBEDDING_MODEL=text-embedding-3-large
```

---

# コアワークフロー

1. **ingest**: GitHub URL or local path → 取得（clone/zip/contents API）。巨大リポジトリは拡張子やサイズでフィルタ。
2. **index**: コード/ドキュメントをチャンク化し埋め込み作成 →FAISS へ格納。
3. **analyze**: 言語別の主要エントリ、依存、ビルド/実行方法、公開 API や CLI を推定。
4. **generate**:

   - `repo_overview.md`（目的/構造/ビルド/実行）
   - `samples/` サンプルコード（Quickstart、主要ユースケース）
   - `docs/README.draft.md`（章立てテンプレ含む）

---

# 主要ファイルの実装例

## src/agent/utils.py

```python
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".rb", ".php"}
DOC_EXTS  = {".md", ".rst", ".txt"}
BIN_EXTS  = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".lock"}

IGNORES = {"node_modules", ".git", ".venv", "dist", "build", "site-packages"}

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in IGNORES:
                continue
        else:
            if p.suffix.lower() in BIN_EXTS:
                continue
            yield p

def is_code(path: Path) -> bool:
    return path.suffix.lower() in CODE_EXTS

def is_doc(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTS
```

## src/agent/ingest.py

```python
from __future__ import annotations
import os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Optional
import requests
from rich import print

class Ingestor:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    def from_github(self, repo_url: str, branch: str|None=None) -> Path:
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
        # 1階層下に展開される想定 owner-name-branch/
        unpacked = next(self.workdir.glob(f"{name}-*"))
        return unpacked
```

## src/agent/index.py

```python
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import faiss, numpy as np
import tiktoken
from .utils import iter_files
from openai import OpenAI
import os

class VectorIndex:
    def __init__(self, model: str, index_path: Path):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.index_path = index_path
        self.index = None
        self.docs: List[str] = []
        self.paths: List[str] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        res = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in res.data], dtype="float32")

    def build(self, root: Path, chunk_size: int = 1200, chunk_overlap: int = 150):
        enc = tiktoken.get_encoding("cl100k_base")
        chunks: List[str] = []
        meta: List[str] = []
        for p in iter_files(root):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            toks = enc.encode(text)
            for i in range(0, len(toks), chunk_size - chunk_overlap):
                sub = enc.decode(toks[i:i+chunk_size])
                chunks.append(sub)
                meta.append(str(p))
        embs = self._embed(chunks)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.docs = chunks
        self.paths = meta

    def search(self, query: str, k: int = 8) -> List[Dict]:
        q = self._embed([query])
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            out.append({"score": float(score), "text": self.docs[idx], "path": self.paths[idx]})
        return out
```

## src/agent/analyze.py

```python
from __future__ import annotations
from pathlib import Path
from collections import Counter
from .utils import iter_files, is_code

LANG_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript/React",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".php": "PHP",
    ".cpp": "C++", ".c": "C",
}

def detect_languages(root: Path):
    exts = [p.suffix for p in iter_files(root) if is_code(p)]
    c = Counter(exts)
    total = sum(c.values()) or 1
    return [{"ext": e, "lang": LANG_MAP.get(e, e), "ratio": n/total} for e, n in c.most_common()]
```

## src/agent/prompts.py

```python
REPO_OVERVIEW_PROMPT = """
You are a senior developer advocate. Summarize the repository for a newcomer.
Return Markdown with sections: Purpose, Key Features, Project Structure, Dependencies, Build & Run, Testing, Gotchas.
Use bullet points and short code blocks. Infer pragmatically if metadata is missing.
"""

SAMPLE_CODE_PROMPT = """
You are a staff engineer. From the context snippets, write a minimal end-to-end Quickstart sample.
Constraints:
- make it runnable in one file
- show dependency install
- show input & output example
- prefer simplest happy path
"""

README_PROMPT = """
You are a technical writer. Draft a production-ready README with:
- Title, Badges (TODO), Overview
- Installation
- Quickstart (copy-paste runnable)
- API Reference (top 3-5 public functions)
- Examples (2 scenarios)
- Architecture Diagram (ASCII if needed)
- Development (tests, lint, format)
- Contributing
- License
Keep concise but complete. Use the project’s actual names when available.
"""
```

## src/agent/generate.py

```python
from __future__ import annotations
from pathlib import Path
from typing import List
from openai import OpenAI
import os, json

class Generator:
    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _chat(self, system: str, user: str):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return res.choices[0].message.content

    def render_overview(self, context_chunks: List[str]) -> str:
        from .prompts import REPO_OVERVIEW_PROMPT
        joined = "\n\n---\n\n".join(context_chunks[:12])
        return self._chat("You write crisp, accurate docs.", REPO_OVERVIEW_PROMPT + "\n\nCONTEXT:\n" + joined)

    def render_sample(self, context_chunks: List[str]) -> str:
        from .prompts import SAMPLE_CODE_PROMPT
        joined = "\n\n---\n\n".join(context_chunks[:12])
        return self._chat("You write runnable code.", SAMPLE_CODE_PROMPT + "\n\nCONTEXT:\n" + joined)

    def render_readme(self, context_chunks: List[str]) -> str:
        from .prompts import README_PROMPT
        joined = "\n\n---\n\n".join(context_chunks[:20])
        return self._chat("You produce production docs.", README_PROMPT + "\n\nCONTEXT:\n" + joined)
```

## src/agent/main.py（CLI）

```python
from __future__ import annotations
import os
from pathlib import Path
import typer
from rich import print
from dotenv import load_dotenv
from .ingest import Ingestor
from .index import VectorIndex
from .analyze import detect_languages
from .generate import Generator

app = typer.Typer()

@app.command()
def run(repo: str = typer.Argument(..., help="GitHub URL or local path"), branch: str|None = None):
    load_dotenv()
    workdir = Path(".work"); workdir.mkdir(exist_ok=True)
    outputs = Path("outputs"); (outputs / "samples").mkdir(parents=True, exist_ok=True); (outputs / "docs").mkdir(parents=True, exist_ok=True)

    # 1) ingest
    ing = Ingestor(workdir)
    root = Path(repo)
    if root.exists():
        print(f"[bold cyan]Using local path:[/] {root}")
    else:
        print(f"[bold cyan]Fetching from GitHub:[/] {repo}")
        root = ing.from_github(repo, branch)

    # 2) index
    idx = VectorIndex(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"), index_path=workdir/"faiss.index")
    idx.build(root)

    # 3) analyze
    langs = detect_languages(root)
    print("[bold]Languages detected:[/]", langs)

    # 4) gather context（代表チャンクを検索して抽出）
    seeds = [
        "README usage install quickstart",
        "setup build run test",
        "public api class function cli",
    ]
    chunks = []
    for q in seeds:
        for hit in idx.search(q, k=5):
            chunks.append(f"FILE: {hit['path']}\n\n{hit['text']}")

    # 5) generate
    gen = Generator(model=os.getenv("MODEL_NAME", "gpt-4.1-mini"))
    overview = gen.render_overview(chunks)
    sample   = gen.render_sample(chunks)
    readme   = gen.render_readme(chunks)

    (outputs/"repo_overview.md").write_text(overview, encoding="utf-8")
    (outputs/"samples"/"quickstart.py").write_text(sample, encoding="utf-8")
    (outputs/"docs"/"README.draft.md").write_text(readme, encoding="utf-8")

    print("\n[green]Done.[/]")
    print(" - outputs/repo_overview.md")
    print(" - outputs/samples/quickstart.py")
    print(" - outputs/docs/README.draft.md")

if __name__ == "__main__":
    app()
```

---

# 生成ドキュメントの章立て（テンプレ）

`README.draft.md` は以下の章立てで生成されます。

````markdown
# {Project Title}

## Overview

- What it does / Why it exists

## Installation

- Requirements
- Install commands

## Quickstart

```bash
# copy & run
```
````

## API Reference (Top functions)

- signature, params, returns, example

## Examples

- Example A
- Example B

## Architecture

```text
# ASCII diagram
```

## Development

- repo layout, tasks, tests, lint, fmt

## Contributing / License

````

---

# 実行例
```bash
uv sync
cp .env.example .env  # APIキー設定
python -m src.agent.main https://github.com/{owner}/{repo}
````

---

# 拡張アイデア

- **LangGraph** でステップ分岐（例：README が見つからない → 優先探索）。
- **Guardrails**: 生成コードを `pytest --run-samples` でスモーク実行し、失敗時は自動再生成。
- **AST 解析**: `tree_sitter` で公開 API 抽出を精緻化。
- **マルチモーダル**: アーキテクチャ図をテキスト説明 → 画像生成に差し替え。
- **キャッシュ**: 既存埋め込みと差分更新（コミット範囲のみ再計算）。
- **社内導入**: Private repo は `GITHUB_TOKEN` でアクセス、成果物は S3/Cloud Storage へ保存。
- **日本語/英語切替**: CLI フラグ `--lang ja|en`。

---

# セキュリティと運用

- 機密コードを外部モデルへ送る際は **同意・社内規定** を順守。オンプレ/自社モデルの選択肢も用意。
- 生成物に **自動ウォーターマーク**（生成日時・コミット SHA・モデル名）を付与。
- レート制御：GitHub API は条件付きリクエストと ETag 活用。

---

# 次のステップ（あなたの環境向け）

1. リポジトリ URL の例を 1 つ教えてください（公開でも私物でも OK）。
2. 上のスターターをそのリポジトリに最適化（言語別ルール/サンプル）します。
3. 必要なら **LangGraph 版ワークフロー** と **Streamlit UI** を追加します。
