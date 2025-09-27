from pathlib import Path

from tech_post_agent.model import chat_model
from tech_post_agent.tools.core.index import load_or_create_index


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    repo_dir = root_dir / ".cache/work/github/ingest/openpoke-main"

    index = load_or_create_index(root=repo_dir)
    model = chat_model("gpt-4o-mini")
    qe = index.as_query_engine(llm=model)
    resp = qe.query("このリポの目的は？")

    print(resp)


if __name__ == "__main__":
    main()
