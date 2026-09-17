import os
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX = "faiss_index"
DATA_DIR = "data"

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
    
    print(f"{len(chunks)} fragments créés. Initialisation du réseau d'embedding sur le CPU...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Vectorisation et création de la base de données FAISS...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX)
    print(f"\n Succès ! La base de données a été sauvegardée dans '{FAISS_INDEX}'.")

if __name__ == "__main__":
    ingest()
