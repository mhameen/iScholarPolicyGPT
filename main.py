from fastapi import FastAPI
from pydantic import BaseModel
import traceback

app = FastAPI()

qa_chain = None  # global

class QueryRequest(BaseModel):
    query: str

# ✅ Health check
@app.get("/")
def home():
    return {"status": "running"}

# ✅ Lazy load function
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
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        print("RAG Loaded ✅")

    return qa_chain


# ✅ API endpoint
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        chain = get_qa_chain()

        response = chain.invoke({"query": req.query})

        return {"answer": response["result"]}

    except Exception as e:
        return {
            "error": repr(e),
            "trace": traceback.format_exc()
        }