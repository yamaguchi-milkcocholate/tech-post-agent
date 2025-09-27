from langgraph.graph import MessagesState
from pydantic import BaseModel


class Repo(BaseModel):
    name: str
    url: str
    local_path: str


class State(MessagesState):
    # This state class has the messages key build in
    pass
