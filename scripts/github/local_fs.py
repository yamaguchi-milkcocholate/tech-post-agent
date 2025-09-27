import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from tech_post_agent.mcp import filesystem_mcp
from tech_post_agent.model import chat_model


async def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    fs_dir = root_dir / ".cache/work/github"

    # データディレクトリが存在しない場合は作成
    fs_dir.mkdir(parents=True, exist_ok=True)

    # Docker MCPを使用する設定
    servers_config = {}
    servers_config.update(filesystem_mcp("filesystem", str(fs_dir)))

    # MultiServerMCPClientに設定を渡す
    client = MultiServerMCPClient(servers_config)
    tools = await client.get_tools()  # Filesystem MCP のツール群をロード

    model = chat_model()
    agent = create_react_agent(model, tools)

    # 例: ディレクトリ一覧やファイル読み取り（ツール名は実装により "list_dir"/"read_file" など）
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "data/ の内容を一覧して"}]}
    )
    for m in response["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
