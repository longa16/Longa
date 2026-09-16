import os
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_REPO_ID    = "meta-llama/Llama-3.1-8B-Instruct"
FAISS_INDEX    = "faiss_index"


def process_pdf(pdf_path: str) -> list:
    """Charge un PDF et le découpe en chunks."""
    print(f"Chargement du fichier : {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"Fichier introuvable : {pdf_path}")
        return []

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunking terminé : {len(chunks)} chunks créés.")
    return chunks


def create_vector_db(chunks: list):
    """Crée et sauvegarde l'index FAISS à partir des chunks."""
    print("Création de la base vectorielle...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX)
    print("Index FAISS sauvegardé.")
    return db


def load_rag_chain():
    """Charge l'index FAISS et construit la chaîne RAG."""
    print(f"Chargement du modèle : {LLM_REPO_ID}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        provider="featherless-ai",
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.2,
        max_new_tokens=512,
    )
    chat_llm = ChatHuggingFace(llm=llm)

    prompt_template = """Tu es un assistant expert en analyse de documents. Réponds en français, de manière claire, complète et structurée.
                    Utilise les informations du contexte ci-dessous pour répondre. 
                    Si la réponse ne s'y trouve pas, dis-le clairement.

---
CONTEXTE :
{context}
---

QUESTION : {question}

RÉPONSE DÉTAILLÉE :"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain
