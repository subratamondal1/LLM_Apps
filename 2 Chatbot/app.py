from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models.google_palm import ChatGooglePalm
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

load_dotenv()

chat_model = ChatGooglePalm(verbose=True)

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory(file_path="history.json"),
    memory_key="history",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "history"],
    messages=[
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(template="{content}"),
    ],
)

chain = LLMChain(prompt=prompt, llm=chat_model, memory=memory, verbose=True)

while True:
    content = input(">> Enter...")
    print(f"USER: {content}")
    result = chain({"content": content})

    print(f"AI:{result['text']}")