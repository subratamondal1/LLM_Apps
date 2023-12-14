from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.vectorstores.chroma import Chroma

embeddings = GooglePalmEmbeddings()

# embedded_text = embeddings.embed_query(
#     text = "Hi, I am Subrata Mondal"
# )
# print(len(embedded_text))

# Initialize the TextLoader
loader = TextLoader(
    file_path = "3 facts/facts.txt"
)

# Initialize CharacterTextSplitter
character_text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200, # First it will collect 200 texts and then the separator will be applied
    chunk_overlap = 0
)

# Load the documents
docs = loader.load_and_split(
    text_splitter = character_text_splitter
)

db = Chroma.from_documents(
    documents=docs, # list of tokens
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search_with_score(
    query="How many people does lightning kills every year?"
)

print(len(results))

for result in results:
    print()
    print(result)
    print()
    print(result[0].page_content)
    print()
    print(result[1])
    break

