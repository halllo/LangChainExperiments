from langchain_openai import ChatOpenAI;
from langchain.schema import SystemMessage;
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder;
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_functions_agent;
from langchain.globals import set_debug;
from dotenv import load_dotenv;
from tools.sql import run_query_tool, list_tables, describe_tables_tool;

load_dotenv();

#set_debug(True);

chat = ChatOpenAI();

tables = list_tables();
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist or what columns exist. "
            "Instead, use the 'describe_tables' functions.\n"
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
);

tools = [run_query_tool, describe_tables_tool];

agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools,
);

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
);

agent_executor.invoke({
    "input": "How many users have provided a shipping address?"
});