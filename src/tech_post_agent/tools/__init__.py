# github.pyの@toolデコレータ付き関数・クラスをまとめてimport

from .base import Done, Question
from .github import fetch_trend_repo

__all__ = ["fetch_trend_repo", "Question", "Done"]

tools = [globals()[name] for name in __all__]
tools_by_name = {tool.name: tool for tool in tools}

hmtl_tools = ["Question"]
