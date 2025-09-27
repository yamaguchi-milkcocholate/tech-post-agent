from .github import download_github_repo

__all__ = ["download_github_repo"]

tools = [globals()[name] for name in __all__]
tools_by_name = {tool.name: tool for tool in tools}
