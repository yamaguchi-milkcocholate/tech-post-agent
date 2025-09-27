import asyncio

from tech_post_agent.graphs import super_graph


async def main() -> None:
    agent = await super_graph()

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "openpoke-mainの内容を一覧して",
                    # "content": "このリポジトリについて教えて: https://github.com/shlokkhemani/openpoke",
                }
            ]
        }
    )

    for m in response["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
