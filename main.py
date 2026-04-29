from fastapi import FastAPI
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

qa_chain = None

class QueryRequest(BaseModel):
    query: str


# ✅ HEALTH CHECK
@app.get("/")
def home():
    return {"status": "running"}


# ✅ LAZY LOAD RAG (FIXED)
def get_qa_chain():
    global qa_chain

    if qa_chain is None:
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
            retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True
        )

        print("RAG Loaded Successfully ✅")

    return qa_chain


# ✅ API ENDPOINT (SAFE)
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