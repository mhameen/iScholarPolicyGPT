from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
import os
from dotenv import load_dotenv
load_dotenv()

if os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is set")
else:
    raise ValueError("OPENAI_API_KEY is not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Load your documents
loader = DirectoryLoader(
    "./IKSPL_Emp_Policies",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Split
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Use OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Create FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Save
vectorstore.save_local("faiss_index")

print("FAISS rebuilt successfully ✅")