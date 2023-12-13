from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.google_palm import GooglePalmEmbeddings

embeddings = GooglePalmEmbeddings()
embedded_text = embeddings.embed_query(
    text = "Hi, I am Subrata Mondal"
)
print(len(embedded_text), embedded_text)

# # Initialize CharacterTextSplitter
# character_text_splitter = CharacterTextSplitter(
#     separator = "\n",
#     chunk_size = 200, # First it will collect 200 texts and then the separator will be applied
#     chunk_overlap = 0
# )

# # Initialize the TextLoader
# loader = TextLoader(
#     file_path = "3 facts/facts.txt"
# )

# # Load the documents
# docs = loader.load_and_split(
#     text_splitter = character_text_splitter
# )

# for doc in docs:
#     print(doc)
#     print()