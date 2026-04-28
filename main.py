from fastapi import FastAPI
from pydantic import BaseModel
import traceback

# Your existing imports
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

import os 
from dotenv import load_dotenv
load_dotenv()

import os
print("API KEY:", os.getenv("OPENAI_API_KEY"))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create embeddings (same as before)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Load FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index dimension:", vectorstore.index.d)
print("Embedding dimension:", len(embeddings.embed_query("test")))

# ---- INIT APP ----
app = FastAPI()

# ---- REQUEST FORMAT ----
class QueryRequest(BaseModel):
    query: str

# ---- LOAD YOUR RAG (already built) ----
llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ---- API ENDPOINT ----
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        response = qa_chain.invoke({"query": req.query})
        return {
            "answer": response["result"]
        }
    except Exception as e:
        return {
            "error": repr(e),
            "trace": traceback.format_exc()
        }

# ---- ADD THIS AT THE VERY BOTTOM ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)