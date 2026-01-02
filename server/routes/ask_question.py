from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import Field
import os

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger

router = APIRouter()

# ===================== EMBEDDING MODEL =====================
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User question: {question}")

        # ===================== PINECONE QUERY =====================
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        embedded_query = embed_model.embed_query(question)

        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        if not res.get("matches"):
            return {
                "response": "I'm sorry, but I couldn't find relevant information in the provided documents.",
                "sources": []
            }

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        # ===================== SIMPLE RETRIEVER =====================
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)

        # ===================== LLM CHAIN =====================
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("Query processed successfully")
        return result

    except Exception as e:
        logger.exception("Error while answering question")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
