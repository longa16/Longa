import os
import re
import unicodedata
import warnings
from typing import Any, List, Optional

warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FAISS_INDEX = "faiss_index"

PROMPT_TEMPLATE = """Tu es Longa, l'assistant de Loïc NGASSA. Tu réponds aux recruteurs sur son parcours, ses compétences, ses projets et sa formation.

Règles :
- Réponds uniquement à partir du CONTEXTE. N'invente, ne suppose et ne déduis rien sur Loïc, et évite les formulations hypothétiques ("probablement", "il semble", "cela suggère").
- Si l'information est absente du contexte, réponds exactement : "Je ne dispose pas de cette information dans les documents qui m'ont été fournis et je préfère ne pas spéculer."
- Compétences : distingue ce qui est explicitement déclaré, les technologies explicitement utilisées et les projets associés. Une expérience n'est pas une maîtrise : cite le projet ou l'expérience d'origine.
- Si le recruteur te salue ou demande ce que tu peux faire, réponds poliment en te présentant brièvement.
- Privilégie la précision à la quantité. Réponds en français, sauf si le recruteur écrit dans une autre langue.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """Charge le modèle d'embedding une seule fois et le met en cache."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


@st.cache_resource(show_spinner="Chargement du reranker...")
def get_cross_encoder():
    """Charge le cross-encoder une seule fois."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def tokenize(text: str) -> List[str]:
    """Tokenisation BM25 en minuscules, sans accents, sans ponctuation."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


class HybridRerankRetriever(BaseRetriever):
    """Retriever hybride avec reranking par cross-encoder."""

    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    cross_encoder: Any
    top_k: int = 6
    min_score: Optional[float] = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = self.vector_retriever.invoke(query) + self.bm25_retriever.invoke(query)

        # Dédoublonnage sur le contenu
        unique = list({doc.page_content: doc for doc in reversed(candidates)}.values())
        unique.reverse()
        if not unique:
            return []

        scores = self.cross_encoder.predict([[query, d.page_content] for d in unique])
        ranked = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)

        if self.min_score is not None:
            ranked = [(d, s) for d, s in ranked if s >= self.min_score]
        return [doc for doc, _ in ranked[: self.top_k]]


@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """Charge l'index FAISS et construit la chaîne RAG."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY manquante dans le .env")

    db = FAISS.load_local(FAISS_INDEX, get_embeddings_model(), allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=2048,
        max_retries=1,
        timeout=30,
    )

    vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})

    all_docs = list(db.docstore._dict.values())  # API privée de LangChain
    bm25_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=tokenize)
    bm25_retriever.k = 10

    retriever = HybridRerankRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        cross_encoder=get_cross_encoder(),
        top_k=6,
    )

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )