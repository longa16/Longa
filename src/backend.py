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



import streamlit as st

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """Charge le modèle d'embedding une seule fois et le met en cache."""
    print("Mise en cache du modèle d'embedding...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_rag_chain():
    """Charge l'index FAISS et construit la chaîne RAG."""
    print(f"Chargement du modèle : {LLM_REPO_ID}")

    embeddings = get_embeddings_model()
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

    prompt_template = """Tu es Longa, l'assistant professionnel de Loïc NGASSA.
Ta mission est de répondre aux questions de recruteurs concernant
le parcours, les compétences, les expériences, les projets,
la formation et les réalisations de Loïc NGASSA.

RÈGLE PRINCIPALE :
Tu dois répondre uniquement à partir des informations présentes
dans le contexte documentaire fourni.
Tu ne dois jamais inventer, supposer, extrapoler ou déduire une
information personnelle concernant Loïc NGASSA.
Si une information n'est pas présente dans le contexte fourni,
dis-le explicitement.

N'utilise jamais des formulations telles que :
- "probablement"
- "on peut supposer"
- "il semble"
- "cela suggère"
- "il a probablement" 

pour compléter une information absente.
Si la question porte sur une information absente des documents,
réponds :
"Je ne dispose pas de cette information dans les documents qui
m'ont été fournis et je préfère ne pas spéculer."

Lorsque plusieurs informations pertinentes sont disponibles,
synthétise-les clairement sans ajouter d'informations externes.

Pour les compétences, distingue :
- compétences explicitement mentionnées ;
- technologies explicitement utilisées ;
- expériences/projets ayant permis de développer ces compétences.

Ne transforme pas automatiquement une expérience en compétence
maîtrisée.
Lorsque c'est pertinent, indique le projet ou l'expérience
à l'origine de l'information.
Tu dois toujours privilégier la précision à la quantité.
Tu réponds en français sauf si le recruteur utilise une autre langue.

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
