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

# ✅ Now get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

qa_chain = None


class QueryRequest(BaseModel):
    query: str


# ✅ HEALTH CHECK
@app.get("/")
def home():
    return {"status": "running"}


# ✅ LAZY LOAD RAG
def get_qa_chain():
    global qa_chain

    if qa_chain is None:
        print("Loading RAG...")

        # ✅ SAME embeddings as FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # ✅ 🔥 FIX: Absolute path (Railway compatible)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(BASE_DIR, "faiss_index")

        print("FAISS PATH:", faiss_path)

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
        print("Embedding dimension:", len(embeddings.embed_query("test")))

    return qa_chain


# ✅ API ENDPOINT
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