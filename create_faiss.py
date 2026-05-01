from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# ✅ Load documents
loader = DirectoryLoader(
    "./IKSPL_Emp_Policies",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# ✅ Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# ✅ Use HuggingFace embeddings ONLY
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ Create FAISS (ONLY ONCE)
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Save FAISS
vectorstore.save_local("faiss_index")

print("FAISS rebuilt correctly ✅")