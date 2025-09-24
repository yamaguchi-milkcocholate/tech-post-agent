import asyncio

import streamlit as st

from tech_post_agent.graphs.react import agent


async def main():
    st.title("技術ブログチャットボット")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # チャットボットの応答
        with st.chat_message("assistant"):
            expander = st.expander("tool use proccess")
            message_placeholder = st.empty()
            contents = ""
            tool_outputs = []

            async for event in agent.astream_events(
                {"messages": st.session_state.messages}, version="v1"
            ):
                # メッセージの表示
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        contents += content
                        message_placeholder.markdown(contents)
                # ツール利用の開始
                elif event["event"] == "on_tool_start":
                    tmp = f"#### Start using the tool ： {event['name']}  \nInputs: {event['data'].get('input')}"
                    tool_outputs.append(tmp)
                    expander.markdown(tmp)
                # ツール利用の終了
                elif event["event"] == "on_tool_end":
                    tmp = f"#### Finish using the tool ： {event['name']}  \nOutput ： {event['data'].get('output')}"
                    tool_outputs.append(tmp)
                    expander.markdown(tmp)

            st.session_state.messages.append(
                {"role": "assistant", "content": contents, "tools": tool_outputs}
            )


if __name__ == "__main__":
    asyncio.run(main())
