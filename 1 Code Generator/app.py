from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.google_palm import GooglePalm
from langchain.prompts import PromptTemplate

load_dotenv()  # Will automatically load the Environment Variables from the .env file

LLM = GooglePalm()  # LLM Model

# The template expects two variables: 'language' and 'task'
prompt_template1 = PromptTemplate(
    template="Write a {language} function, that {task}",
    input_variables=["language", "task"],
)

prompt_template2 = PromptTemplate(
    template="Write a test with pytest to test the code: {code}",
    input_variables=["code"],
)

chain1 = LLMChain(llm=LLM, prompt=prompt_template1, output_key="code")

chain2 = LLMChain(llm=LLM, prompt=prompt_template2, output_key="test")

chains = SequentialChain(
    chains=[chain1, chain2],
    input_variables=[
        "language",
        "task",
    ],  # List of Input Variables required for the first chain
    output_variables=[
        "code",
        "test",
    ],  # List of Output Variables expected from the last chain
)

result = chains(
    inputs={"language": "Python", "task": "generate list of even numbers from 1 to 10."}
)

print(">>>Generate Result<<<")
print(result)
print(">>>Generate Code<<<")
print(result["code"])
print(">>>Generate Test<<<")
print(result["test"])
