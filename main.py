from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

app = FastAPI()

# ---- GLOBAL ----
qa_chain = None


# ---- REQUEST MODEL ----
class QueryRequest(BaseModel):
    query: str


# ---- HEALTH CHECK ----
@app.get("/")
def home():
    return {"status": "running"}


# ---- LAZY LOAD RAG ----
def get_qa_chain():
    global qa_chain

    if qa_chain is None:
        print("STEP 1: Starting RAG load")

        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain_classic.chains import RetrievalQA

        print("STEP 2: Imports done")

        # ✅ Use OpenAI embeddings (fast, no heavy download)
        embeddings = OpenAIEmbeddings()
        print("STEP 3: Embeddings ready")

        # ✅ Load FAISS index
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("STEP 4: FAISS loaded")

        # ✅ LLM
        llm = ChatOpenAI(model="gpt-4o-mini")
        print("STEP 5: LLM ready")

        # ✅ Optimize retrieval (IMPORTANT)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )

        print("STEP 6: RAG ready ✅")

    return qa_chain


# ---- MAIN ENDPOINT ----
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        print("Incoming query:", req.query)

        chain = get_qa_chain()

        response = chain.invoke({"query": req.query})

        print("Response generated")

        return {
            "answer": response["result"]
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


# ---- DEBUG ROUTE ----
@app.get("/test")
def test():
    return {"message": "API working fine"}