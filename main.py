from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ---- GLOBAL VARIABLE ----
qa_chain = None

# ---- REQUEST MODEL ----
class QueryRequest(BaseModel):
    query: str

# ---- HEALTH CHECK (VERY IMPORTANT) ----
@app.get("/")
def home():
    return {"status": "running"}

# ---- LOAD RAG AFTER SERVER STARTS ----
@app.on_event("startup")
def load_rag():
    global qa_chain

    print("Loading RAG...")

    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_classic.chains import RetrievalQA

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    print("RAG Loaded Successfully!")

# ---- API ENDPOINT ----
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        if qa_chain is None:
            return {"error": "Model is still loading. Try again in a few seconds."}

        response = qa_chain.invoke({"query": req.query})

        return {
            "answer": response["result"]
        }

    except Exception as e:
        return {
            "error": repr(e),
            "trace": traceback.format_exc()
        }