from langchain_core.tools import tool
from pydantic import BaseModel


@tool
class Question(BaseModel):
    """Question to ask user."""

    content: str


@tool
class Done(BaseModel):
    """Process has been done."""

    done: bool
