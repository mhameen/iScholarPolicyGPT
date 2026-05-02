from fastapi import FastAPI
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv
import os

# ✅ Load env FIRST
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 🔥 NEW imports for rebuilding FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ✅ Now get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
qa_chain = None


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "running"}


def get_qa_chain():
    global qa_chain

    if qa_chain is None:
        print("Loading RAG...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(BASE_DIR, "faiss_index")

        index_file = os.path.join(faiss_path, "index.faiss")
        pkl_file = os.path.join(faiss_path, "index.pkl")

        print("FAISS PATH:", faiss_path)

        # ✅ 🔥 CORRECT CHECK
        if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
            print("FAISS missing or incomplete → rebuilding inside Railway...")

            # ⚠️ CHANGE THIS to your actual file
            loader = TextLoader("your_file.txt")
            documents = loader.load()

            splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = splitter.split_documents(documents)

            db = FAISS.from_documents(docs, embeddings)
            db.save_local(faiss_path)

            print("FAISS built successfully ✅")

        # ✅ Load FAISS
        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True
        )

        print("RAG Loaded Successfully ✅")

    return qa_chain


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        chain = get_qa_chain()
        response = chain.invoke({"query": req.query})

        return {"answer": response["result"]}

    except Exception as e:
        print("ERROR:", e)
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }