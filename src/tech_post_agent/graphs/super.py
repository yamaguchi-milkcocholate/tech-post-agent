from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from tech_post_agent.mcp import filesystem_mcp
from tech_post_agent.model import chat_model


async def super_graph() -> CompiledStateGraph:
    from tech_post_agent.prompts import super_agent_system_prompt
    from tech_post_agent.tools import download_github_repo
    from tech_post_agent.tools.prompts import tool_prompt

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    fs_dir = root_dir / ".cache/work/github/ingest"

    servers_config = {}
    servers_config.update(filesystem_mcp("filesystem", str(fs_dir)))
    client = MultiServerMCPClient(servers_config)
    mcp_tools = await client.get_tools()  # Filesystem MCP のツール群をロード

    llm = chat_model("gpt-4o-mini")
    agent = create_react_agent(
        llm,
        tools=[download_github_repo] + mcp_tools,
        prompt=super_agent_system_prompt.format(tools_prompt=tool_prompt),
    )

    return agent
