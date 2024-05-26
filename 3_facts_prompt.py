from langchain_openai import ChatOpenAI;
from langchain_openai.embeddings import OpenAIEmbeddings;
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder;
from langchain_community.vectorstores.chroma import Chroma;
from langchain.chains.retrieval import create_retrieval_chain;
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv;

load_dotenv();


chat = ChatOpenAI();
embeddings = OpenAIEmbeddings();

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
);

retriever = db.as_retriever()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
);
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
);

qa_chain = create_stuff_documents_chain(llm=chat, prompt=prompt);
chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain);

result = chain.invoke({
    "input": "What is an iteresting fact about the English language?"
});

print(result["answer"]);
