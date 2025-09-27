from tech_post_agent.graphs import super_graph


def main() -> None:
    agent = super_graph()

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "このリポジトリについて教えて: https://github.com/shlokkhemani/openpoke",
                }
            ]
        }
    )

    for m in response["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
