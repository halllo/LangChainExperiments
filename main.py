from langchain_openai import OpenAI;
from langchain.prompts import PromptTemplate;
from langchain_core.output_parsers import StrOutputParser;
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick ;
from langchain.chains.sequential import SequentialChain;
from operator import itemgetter
import argparse;
from dotenv import load_dotenv;


parser = argparse.ArgumentParser(description='Process some integers.');
parser.add_argument('--task', default='return a list of numbers');
parser.add_argument('--language', default='python');
args = parser.parse_args();

load_dotenv();


llm = OpenAI();

code_prompt = PromptTemplate.from_template("Write a very short {language} function that will {task}");
code_chain = code_prompt | llm | {"code": StrOutputParser()};

test_prompt = PromptTemplate.from_template("Write a test for the following {language} code:\n{code}");
test_chain = test_prompt | llm | {"test": StrOutputParser()};

chain = code_chain | test_chain;

print(chain.input_schema.schema())
print(chain.output_schema.schema())

result = chain.invoke({
    "language": args.language,
    "task": args.task,
});

print(result);