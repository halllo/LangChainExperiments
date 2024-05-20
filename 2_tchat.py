from langchain_openai import ChatOpenAI;
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder;
from langchain_core.runnables.history import RunnableWithMessageHistory;
from langchain.memory import FileChatMessageHistory, ConversationSummaryMemory;
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory;
from dotenv import load_dotenv;

load_dotenv();

chat = ChatOpenAI();

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}"),
])

chain = prompt | chat;

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # return ConversationSummaryMemory( # does not work, because ConversationSummaryMemory is not a subclass of BaseChatMessageHistory
    #     llm=chat,
    #     chat_memory=FileChatMessageHistory(f"messages_{session_id}.json"),
    #     input_key="content",
    #     memory_key="messages",
    #     return_messages=True,
    # );
    return FileChatMessageHistory(f"messages_{session_id}.json");

with_message_history = RunnableWithMessageHistory(
    chain, 
    get_session_history=get_session_history,
    input_messages_key="content",
    history_messages_key="messages",
);

while True:
    content = input(">> ");
    result = with_message_history.invoke(
        input={
            "content": content,
        },
        config={
            "configurable": {"session_id": "abc123"}
        }
    );
    print(result.content);