from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tech_post_agent.prompts import agent_system_prompt, tool_prompt
from tech_post_agent.schema import State
from tech_post_agent.tools import tools, tools_by_name

load_dotenv()

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")


def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    result = llm_with_tools.invoke(
        # Add the system prompt
        [
            {
                "role": "system",
                "content": agent_system_prompt.format(tools_prompt=tool_prompt),
            }
        ]
        # Add the current messages to the prompt
        + state["messages"]
    )
    print(result)
    return {"messages": [result]}


def _tool_invoke(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return {
        "role": "tool",
        "content": observation,
        "tool_call_id": tool_call["id"],
    }


# ツール実行ノード
def tool_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    result = []
    goto = "llm_call"
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = _tool_invoke(tool_call)
        result.append(tool_result)
        if tool_call["name"] == "Question":
            # result.append({"role": "ai", "content": observation.content})
            goto = END  # ループを止めたい場合
    update = {"messages": result}
    return Command(goto=goto, update=update)


def should_continue(state: State) -> Literal["tool_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] in ("Done", "Question"):
                return END
            else:
                return "tool_handler"


"""Main function to demonstrate the use of imported variable."""
# Build workflow
overall_workflow = StateGraph(State)


# Add nodes
overall_workflow.add_node("llm_call", llm_call)
overall_workflow.add_node("tool_handler", tool_handler)


# Add edges
overall_workflow.add_edge(START, "llm_call")
overall_workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_handler": "tool_handler",
        END: END,
    },
)
overall_workflow.add_edge("tool_handler", "llm_call")

# Compile the agent
agent = overall_workflow.compile()
