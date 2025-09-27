from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from tech_post_agent.model import chat_model


def super_graph() -> CompiledStateGraph:
    from tech_post_agent.prompts import super_agent_system_prompt
    from tech_post_agent.tools import download_github_repo
    from tech_post_agent.tools.prompts import tool_prompt

    llm = chat_model("gpt-4o-mini")
    agent = create_react_agent(
        llm,
        tools=[download_github_repo],
        prompt=super_agent_system_prompt.format(tools_prompt=tool_prompt),
    )

    return agent
