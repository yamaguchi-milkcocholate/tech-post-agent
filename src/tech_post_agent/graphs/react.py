from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from tech_post_agent.prompts import agent_system_prompt, tool_prompt
from tech_post_agent.tools import tools

load_dotenv()

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
agent = create_react_agent(
    llm, tools=tools, prompt=agent_system_prompt.format(tools_prompt=tool_prompt)
)
