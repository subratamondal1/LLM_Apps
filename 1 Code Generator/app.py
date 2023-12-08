from langchain.llms.google_palm import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os
from dotenv import load_dotenv

load_dotenv() # Will automatically load the Environment Variables from the .env file

LLM = GooglePalm() # LLM Model

prompt_template1 = PromptTemplate(
    template = "Write a {language} function, that {task}",
    input_variables = ["language", "task"]
)

prompt_template2 = PromptTemplate(
    template = "Write a test with pytest to test the code: {code}",
    input_variables = ["code"]
)

chain1 = LLMChain(
    llm = LLM,
    prompt = prompt_template1,
    output_key = "code"
)

chain2 = LLMChain(
    llm = LLM,
    prompt = prompt_template2,
    output_key = "test"
)

chains = SequentialChain(
    chains = [chain1, chain2],
    input_variables = ["language", "task"],
    output_variables = ["code", "test"]
)

result = chains(
    inputs = {
        "language" : "Python",
        "task" : "generate list of even numbers from 1 to 10."
    }
)

print(">>>Generate Result<<<")
print(result)
print(">>>Generate Code<<<")
print(result["code"])
print(">>>Generate Test<<<")
print(result["test"])
