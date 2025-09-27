from langchain_core.tools import Tool

from tech_post_agent.tools.core.index import RepoIndex


def get_llama_index_tools() -> list[Tool]:
    index = RepoIndex()

    search_tool = Tool.from_function(
        name="search_repo",
        description="search in the repository",
        func=index.search,
    )
    answer_tool = Tool.from_function(
        name="answer_with_context",
        description="generate an answer with context from the repository",
        func=index.answer,
    )
    return [search_tool, answer_tool]
