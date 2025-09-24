import uuid
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tech_post_agent.prompts import agent_system_prompt, tool_prompt
from tech_post_agent.schema import State
from tech_post_agent.tools import hmtl_tools, tools, tools_by_name

load_dotenv()

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")


def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            # Invoke the LLM
            llm_with_tools.invoke(
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
        ]
    }


def _tool_invoke(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return {
        "role": "tool",
        "content": observation,
        "tool_call_id": tool_call["id"],
    }


def interrupt_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    # Store messages
    result = []

    # Go to the LLM call node next
    goto = "llm_call"

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Human-In-the-Loopが必要ないツールはそのままツールを実行
        if tool_call["name"] not in hmtl_tools:
            result.append(_tool_invoke(tool_call))
            continue

        if tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
            description = f"""# Question for User

            {tool_call["args"].get("content")}
            """
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Create the interrupt request
        request = {
            "action_request": {"action": tool_call["name"], "args": tool_call["args"]},
            "config": config,
            "description": description,
        }

        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]

        # Handle the responses
        if response["type"] == "accept":
            # Execute the tool with original args
            result.append(_tool_invoke(tool_call))
        elif response["type"] == "edit":
            raise NotImplementedError("Editing is not implemented yet.")
        elif response["type"] == "ignore":
            # Don't execute the tool, and tell the agent how to proceed
            result.append(
                {
                    "role": "tool",
                    "content": "User ignored this question. End the workflow.",
                    "tool_call_id": tool_call["id"],
                }
            )
            # Go to END
            goto = END
        elif response["type"] == "response":
            user_feedback = response["args"]
            # Don't execute the tool, and add a message with the user feedback to incorporate into the email
            result.append(
                {
                    "role": "tool",
                    "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}",
                    "tool_call_id": tool_call["id"],
                }
            )
        else:
            raise ValueError(f"Invalid response type: {response['type']}")

    # Update the state
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)


def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called."""

    # Get the last message
    messages = state["messages"]
    last_message = messages[-1]

    # Check if it's a Done tool call
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"


def main():
    """Main function to demonstrate the use of imported variable."""
    # Build workflow
    overall_workflow = StateGraph(State)

    # Add nodes
    overall_workflow.add_node("llm_call", llm_call)
    overall_workflow.add_node("interrupt_handler", interrupt_handler)

    # Add edges
    overall_workflow.add_edge(START, "llm_call")
    overall_workflow.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "interrupt_handler": "interrupt_handler",
            END: END,
        },
    )
    overall_workflow.add_edge("interrupt_handler", "llm_call")

    # Compile the agent
    checkpointer = InMemorySaver()
    agent = overall_workflow.compile(checkpointer=checkpointer)
    thread_id_1 = uuid.uuid4()
    thread_config_1 = {"configurable": {"thread_id": thread_id_1}}

    # Run the graph until a tool call that we choose to interrupt
    print("Running the graph until the first interrupt...")
    for chunk in agent.stream({}, config=thread_config_1):
        # Inspect interrupt object if present
        if "__interrupt__" in chunk:
            Interrupt_Object = chunk["__interrupt__"][0]
            print("\nINTERRUPT OBJECT:")
            print(f"Action Request: {Interrupt_Object.value[0]['action_request']}")


if __name__ == "__main__":
    main()
