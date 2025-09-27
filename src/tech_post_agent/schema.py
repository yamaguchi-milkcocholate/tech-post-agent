from langgraph.graph import MessagesState
from pydantic import BaseModel


class Repo(BaseModel):
    name: str
    url: str
    local_path: str


class FileMeta(BaseModel):
    repo_name: str
    name: str
    relative_path: str
    absolute_path: str
    type: str


class File(BaseModel):
    meta: FileMeta
    content: str


class State(MessagesState):
    # This state class has the messages key build in
    pass
