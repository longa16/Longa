import os
import time
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX = "faiss_index"
DATA_DIR = "data"

BATCH = 50    # nombre de chunks par lot
PAUSE = 62    # pause entre lots pour repartir sur une nouvelle fenêtre d'1 minute


def embed_batch_with_retry(db, batch, embeddings, max_retries=6):
    """Indexe un lot ; en cas de 429, attend puis réessaie."""
    for attempt in range(max_retries):
        try:
            if db is None:
                return FAISS.from_documents(batch, embeddings)
            db.add_documents(batch)
            return db
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                wait = 30 * (attempt + 1)
                print(f"  Quota atteint, nouvelle tentative dans {wait}s "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Trop de tentatives échouées sur le quota Gemini.")


def ingest():
    print(f"Chargement des PDFs depuis le répertoire '{DATA_DIR}'...")
    if not os.path.exists(DATA_DIR):
        print(f"Erreur : le dossier {DATA_DIR} n'existe pas.")
        return

    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print("Aucun document PDF trouvé dans le dossier 'data'.")
        return

    print(f"{len(documents)} pages trouvées au total. Découpage en cours...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    print(f"{len(chunks)} fragments créés. Initialisation des embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("Vectorisation et création de la base de données FAISS...")
    db = None
    total = len(chunks)
    for i in range(0, total, BATCH):
        batch = chunks[i:i + BATCH]
        db = embed_batch_with_retry(db, batch, embeddings)
        db.save_local(FAISS_INDEX)  # sauvegarde incrémentale
        print(f"  {min(i + BATCH, total)}/{total} fragments indexés")
        if i + BATCH < total:
            time.sleep(PAUSE)

    print(f"\nSuccès ! La base de données a été sauvegardée dans '{FAISS_INDEX}'.")


if __name__ == "__main__":
    ingest()